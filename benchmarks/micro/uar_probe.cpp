// uar_probe: can the GB10 map an mlx5 send-queue doorbell (UAR/BlueFlame
// page) for GPU access? The precondition of a kernel-rung doorbell. The
// expected answer may be "unsupported"; either way it is a completed probe.
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
int main() {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) { std::printf("UAR-PROBE no RDMA device\n"); return 2; }
  int rc = 1;
  for (int i = 0; i < n; ++i) {
    if (!mlx5dv_is_supported(list[i])) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (!ctx) continue;
    ibv_pd* pd = ibv_alloc_pd(ctx);
    ibv_cq* cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
    ibv_qp_init_attr ia{}; ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC; ia.cap.max_send_wr = 16; ia.cap.max_recv_wr = 16; ia.cap.max_send_sge = 2; ia.cap.max_recv_sge = 2;
    ibv_qp* qp = pd && cq ? ibv_create_qp(pd, &ia) : nullptr;
    if (!qp) { std::printf("UAR-PROBE %s: qp create failed\n", ibv_get_device_name(list[i])); ibv_close_device(ctx); continue; }
    mlx5dv_qp dvqp{}; mlx5dv_obj obj{}; obj.qp.in = qp; obj.qp.out = &dvqp;
    const int r = mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP);
    std::printf("UAR-PROBE %s: init_obj %d, bf.reg %p size %u, dbrec %p, sq.buf %p (%u wqe x %u B)\n", ibv_get_device_name(list[i]), r, dvqp.bf.reg, dvqp.bf.size, (void*)dvqp.dbrec, dvqp.sq.buf, dvqp.sq.wqe_cnt, dvqp.sq.stride);
    if (r == 0 && dvqp.bf.reg) {
      const long page = sysconf(_SC_PAGESIZE);
      void* base = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dvqp.bf.reg) & ~uintptr_t(page - 1));
      for (unsigned flags : {unsigned(cudaHostRegisterIoMemory | cudaHostRegisterMapped), unsigned(cudaHostRegisterIoMemory), unsigned(cudaHostRegisterMapped)}) {
        const cudaError_t e = cudaHostRegister(base, size_t(page), flags);
        std::printf("UAR-PROBE   cudaHostRegister(uar page %p, %ld, flags 0x%x) -> %d (%s)\n", base, page, flags, int(e), cudaGetErrorString(e));
        if (e == cudaSuccess) {
          void* dptr = nullptr; const cudaError_t g = cudaHostGetDevicePointer(&dptr, base, 0);
          std::printf("UAR-PROBE   device pointer %p (%s)\n", dptr, cudaGetErrorString(g));
          cudaHostUnregister(base); rc = 0; break;
        }
        (void)cudaGetLastError();
      }
      // The doorbell RECORD and the SQ buffer are ordinary host memory: can those be registered?
      void* dbpage = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dvqp.dbrec) & ~uintptr_t(page - 1));
      const cudaError_t e2 = cudaHostRegister(dbpage, size_t(page), cudaHostRegisterMapped);
      std::printf("UAR-PROBE   cudaHostRegister(dbrec page) -> %d (%s)\n", int(e2), cudaGetErrorString(e2));
      if (e2 == cudaSuccess) cudaHostUnregister(dbpage);
      (void)cudaGetLastError();
    }
    ibv_destroy_qp(qp); ibv_destroy_cq(cq); ibv_dealloc_pd(pd); ibv_close_device(ctx);
    break;
  }
  ibv_free_device_list(list);
  std::printf("UAR-RESULT %s\n", rc == 0 ? "MAPPABLE" : "UNSUPPORTED");
  return rc;
}
