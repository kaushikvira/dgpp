#include "loaders/releasable_range.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace dgpp {
namespace {

void ok(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  throw std::runtime_error(std::string("releasable range: ") + what + " failed (" +
                           (s != nullptr ? s : "unknown") + ")");
}

CUmemAllocationProp device_prop() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    throw std::runtime_error("releasable range: no current CUDA device");
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

}  // namespace

size_t ReleasableRange::granularity() {
  const CUmemAllocationProp prop = device_prop();
  size_t g = 0;
  ok(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity query");
  if (g == 0) throw std::runtime_error("releasable range: zero granularity");
  return g;
}

ReleasableRange::ReleasableRange(size_t bytes) {
  if (bytes == 0) throw std::invalid_argument("releasable range: zero bytes");
  // The runtime's primary context must be current on this thread before the
  // driver API is used beside it.
  cudaFree(nullptr);
  const CUmemAllocationProp prop = device_prop();
  const size_t g = granularity();
  const size_t size = (bytes + g - 1) / g * g;
  CUdeviceptr va = 0;
  ok(cuMemAddressReserve(&va, size, 0, 0, 0), "address reserve");
  CUmemGenericAllocationHandle handle = 0;
  CUresult r = cuMemCreate(&handle, size, &prop, 0);
  if (r != CUDA_SUCCESS) {
    cuMemAddressFree(va, size);
    ok(r, "physical allocation");
  }
  r = cuMemMap(va, size, 0, handle, 0);
  if (r == CUDA_SUCCESS) {
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(va, size, &access, 1);
    if (r != CUDA_SUCCESS) cuMemUnmap(va, size);
  }
  if (r != CUDA_SUCCESS) {
    cuMemRelease(handle);
    cuMemAddressFree(va, size);
    ok(r, "map");
  }
  va_ = va;
  handle_ = handle;
  bytes_ = bytes;
  reserved_ = size;
  mapped_ = size;
}

void ReleasableRange::release() {
  if (mapped_ == 0) return;
  ok(cuMemUnmap(static_cast<CUdeviceptr>(va_), mapped_), "unmap");
  ok(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle_)), "release");
  mapped_ = 0;
  handle_ = 0;
}

void ReleasableRange::destroy() noexcept {
  if (mapped_ != 0) {
    cuMemUnmap(static_cast<CUdeviceptr>(va_), mapped_);
    cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle_));
  }
  if (reserved_ != 0) cuMemAddressFree(static_cast<CUdeviceptr>(va_), reserved_);
  va_ = 0;
  handle_ = 0;
  bytes_ = reserved_ = mapped_ = 0;
}

ReleasableRange::~ReleasableRange() { destroy(); }

ReleasableRange::ReleasableRange(ReleasableRange&& o) noexcept
    : va_(o.va_), handle_(o.handle_), bytes_(o.bytes_), reserved_(o.reserved_), mapped_(o.mapped_) {
  o.va_ = 0;
  o.handle_ = 0;
  o.bytes_ = o.reserved_ = o.mapped_ = 0;
}

ReleasableRange& ReleasableRange::operator=(ReleasableRange&& o) noexcept {
  if (this != &o) {
    destroy();
    va_ = std::exchange(o.va_, 0);
    handle_ = std::exchange(o.handle_, 0);
    bytes_ = std::exchange(o.bytes_, 0);
    reserved_ = std::exchange(o.reserved_, 0);
    mapped_ = std::exchange(o.mapped_, 0);
  }
  return *this;
}

}  // namespace dgpp
