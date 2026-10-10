// The `tnca` PyTorch device (PrivateUse1): storage that lives in TNCA device
// memory (doc/r40_tasks_and_acceptance.md R40-01).
//
// What is here is what has to be C++: an Allocator whose DataPtr holds a
// DEVICE address -- memory the TNCA runtime allocated in the device heap,
// never dereferenced on the host -- and the metadata operators that make a
// tensor of that storage (empty, empty_strided), view it (as_strided, view,
// _reshape_alias, set_) and resize it. They touch sizes, strides, offsets and
// storages only, the same code the CPU uses, because none of them reads the
// data. Everything that moves or computes data -- copies between host and
// device, fill, a scalar read, every kernel -- is the runtime's, in Python
// (tnca.torch_device), and reaches the device through tnca.runtime.Session.
//
// Memory: the allocator calls back into Python (`set_memory_callbacks`):
// alloc(nbytes, device) -> address, free(address, device). The address is the
// device's own, so `tensor.data_ptr()` is where the bytes are on the device
// and a launch passes it as is. A resize that grows a storage allocates anew
// and copies on the device through the `copy` callback; nothing here ever
// memcpys a device address.
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <atomic>
#include <mutex>

namespace tnca {

namespace py = pybind11;

static std::mutex g_mu;
// Set by set_memory_callbacks. Held by pointers that are never deleted: a
// static py::object would be destroyed during C++ static destruction, after
// the interpreter has finalised, and its decref would touch freed memory.
static py::object* g_alloc = nullptr;
static py::object* g_free = nullptr;
static py::object* g_copy = nullptr;
static std::atomic<bool> g_live{false};      // false after shutdown(): no callbacks
static thread_local c10::DeviceIndex g_current = 0;
static c10::DeviceIndex g_count = 1;

static c10::Device device_of(c10::DeviceIndex i) {
  return c10::Device(c10::DeviceType::PrivateUse1, i);
}

static void raw_free(void* p) {
  if (p == nullptr || !g_live.load()) {
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    (*g_free)(reinterpret_cast<uint64_t>(p));
  } catch (const std::exception& e) {
    // A free that fails must not throw through a destructor; the runtime
    // reports a leak or a busy buffer itself.
    PyErr_Clear();
  }
}

struct TncaAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t nbytes) override {
    c10::DeviceIndex dev = g_current;
    if (nbytes == 0) {
      return {nullptr, nullptr, &raw_free, device_of(dev)};
    }
    TORCH_CHECK(g_live.load(), "tnca: the device runtime is not registered "
                "(tnca.torch_device.register())");
    uint64_t addr = 0;
    std::string why;
    {
      py::gil_scoped_acquire gil;
      try {
        addr = (*g_alloc)(nbytes, static_cast<int>(dev)).cast<uint64_t>();
      } catch (py::error_already_set& e) {
        // The runtime's refusal (a full device heap: "out of device memory")
        // becomes torch's own out-of-memory error with the runtime's words,
        // raised from here rather than unwinding a Python exception through
        // the dispatcher.
        why = e.what();
      }
    }
    TORCH_CHECK_WITH(OutOfMemoryError, why.empty(), "tnca: ", why);
    TORCH_CHECK(addr != 0, "tnca: the runtime returned device address 0");
    void* p = reinterpret_cast<void*>(addr);
    return {p, p, &raw_free, device_of(dev)};
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &raw_free;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    py::gil_scoped_acquire gil;
    (*g_copy)(reinterpret_cast<uint64_t>(dest), reinterpret_cast<uint64_t>(src),
           static_cast<uint64_t>(count));
  }
};

static TncaAllocator g_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

// ---- device guard ---------------------------------------------------------
// One default stream per device: R40-01 launches synchronously; streams and
// events with real ordering are R40-02's.
struct TncaGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;
  c10::DeviceType type() const override { return kType; }
  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = getDevice();
    if (d.index() != old.index()) {
      setDevice(d);
    }
    return old;
  }
  c10::Device getDevice() const override { return device_of(g_current); }
  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.type() == kType, "tnca guard: not a tnca device");
    TORCH_CHECK(d.index() >= 0 && d.index() < g_count, "tnca: device ", (int)d.index(),
                " does not exist (", (int)g_count, " device(s))");
    g_current = d.index();
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override { g_current = d.index(); }
  c10::Stream getStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }
  c10::DeviceIndex deviceCount() const noexcept override { return g_count; }
  // Launches complete before they return (R40-01): an event is always done.
  void record(void** event, const c10::Stream&, const c10::DeviceIndex,
              const c10::EventFlag) const override {}
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void*) const override { return true; }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override {}
  void synchronizeDevice(const c10::DeviceIndex) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, TncaGuardImpl);

// ---- metadata operators -----------------------------------------------------
static constexpr c10::DispatchKeySet kKeys{c10::DispatchKey::PrivateUse1};

at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout,
                               std::optional<at::Device> device, std::optional<bool> pin,
                               std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "tnca: only strided tensors");
  c10::DeviceGuard g(device.value_or(device_of(g_current)));
  return at::detail::empty_generic(size, &g_allocator, kKeys,
                                   c10::dtype_or_default(dtype), memory_format);
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                         std::optional<at::Device> device, std::optional<bool> pin) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "tnca: only strided tensors");
  c10::DeviceGuard g(device.value_or(device_of(g_current)));
  return at::detail::empty_strided_generic(size, stride, &g_allocator, kKeys,
                                           c10::dtype_or_default(dtype));
}

at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
                      std::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor reshape_alias(const at::Tensor& self, c10::SymIntArrayRef size,
                         c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size),
                                    C10_AS_INTARRAYREF_SLOW(stride));
}

// Grow (never shrink) a storage on the device: a new allocation, the old
// bytes copied over by the runtime, the storage pointed at the new memory.
static void grow_storage(const c10::Storage& storage, size_t nbytes) {
  if (nbytes <= storage.nbytes()) {
    return;
  }
  c10::DataPtr fresh = g_allocator.allocate(nbytes);
  if (storage.nbytes() > 0 && storage.data() != nullptr) {
    g_allocator.copy_data(fresh.get(), storage.data(), storage.nbytes());
  }
  storage.unsafeGetStorageImpl()->set_data_ptr_noswap(std::move(fresh));
  storage.unsafeGetStorageImpl()->set_nbytes(nbytes);
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size,
                          std::optional<at::MemoryFormat> memory_format) {
  auto* impl = self.unsafeGetTensorImpl();
  int64_t old_numel = impl->numel();
  impl->set_sizes_contiguous(size);
  if (memory_format && *memory_format != at::MemoryFormat::Preserve) {
    impl->empty_tensor_restride(*memory_format);
  }
  int64_t numel = impl->numel();
  if (numel != old_numel || numel > 0) {
    size_t need = at::detail::computeStorageNbytes(size, impl->strides(), self.dtype().itemsize(),
                                                   impl->storage_offset());
    grow_storage(impl->storage(), need);
  }
  return self;
}

at::Tensor& set_source_storage(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_storage_offset(at::Tensor& self, at::Storage source,
                                      c10::SymInt storage_offset, c10::SymIntArrayRef size,
                                      c10::SymIntArrayRef stride) {
  // As set_storage_cpu_, but a storage too small is grown on the device
  // rather than by the CPU's resize (which would memcpy device addresses).
  auto off = storage_offset.expect_int();
  auto sz = C10_AS_INTARRAYREF_SLOW(size);
  auto st = C10_AS_INTARRAYREF_SLOW(stride);
  at::native::checkSetStorage(self, source, off, sz, st);
  self.unsafeGetTensorImpl()->set_storage_offset(off);
  std::optional<at::IntArrayRef> st_opt = st.data() != nullptr ? std::optional<at::IntArrayRef>(st)
                                                                : std::nullopt;
  self.unsafeGetTensorImpl()->set_sizes_and_strides(sz, st_opt ? *st_opt : c10::contiguous_strides(sz));
  size_t need = at::detail::computeStorageNbytes(sz, self.strides(), self.dtype().itemsize(), off);
  grow_storage(self.unsafeGetTensorImpl()->storage(), need);
  return self;
}

// ---- scaled_dot_product_attention: the backend the CPU would choose ----------
// SDPA asks `_fused_sdp_choice`; for a PrivateUse1 tensor its Meta kernel (the
// one tracing runs) asks this stub. The answer is the CPU's own selector on
// the same shapes, dtypes and flags, so a `tnca` SDPA takes exactly the CPU
// path's route: flash-attention-for-CPU where the CPU would, which Inductor
// decomposes into the math it generates Triton for (tnca.inductor), and the
// math fallback otherwise. A device SDPA kernel is R40-08's.
static int64_t fused_sdp_choice(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const std::optional<at::Tensor>& attn_mask, double dropout_p,
                                bool is_causal, std::optional<double> scale, bool enable_gqa) {
  return at::native::_fused_sdp_choice_cpp(q, k, v, attn_mask, dropout_p, is_causal, scale,
                                           enable_gqa);
}

}  // namespace tnca

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &tnca::fused_sdp_choice)
}  // namespace at::native

namespace tnca {

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided", empty_strided);
  m.impl("as_strided", as_strided);
  m.impl("view", view);
  m.impl("_reshape_alias", reshape_alias);
  m.impl("resize_", resize_);
  m.impl("set_.source_Storage", set_source_storage);
  m.impl("set_.source_Storage_storage_offset", set_source_storage_offset);
}

// ---- Python ---------------------------------------------------------------
static void set_memory_callbacks(py::object alloc, py::object free, py::object copy) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_alloc = new py::object(std::move(alloc));
  g_free = new py::object(std::move(free));
  g_copy = new py::object(std::move(copy));
  g_live.store(true);
}

static void shutdown() {
  // At interpreter exit tensors may outlive the runtime; their storages are
  // abandoned with the device rather than freed through a dead callback.
  g_live.store(false);
}

static void set_device_count(int n) {
  TORCH_CHECK(n >= 1 && n < 64, "tnca: device count ", n);
  g_count = static_cast<c10::DeviceIndex>(n);
}

static uint64_t storage_address(const at::Tensor& t) {
  return reinterpret_cast<uint64_t>(t.storage().data());
}

}  // namespace tnca

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "The tnca PyTorch device: device-memory storage and the metadata operators";
  m.def("set_memory_callbacks", &tnca::set_memory_callbacks,
        "alloc(nbytes, device) -> address; free(address); copy(dst, src, nbytes)");
  m.def("shutdown", &tnca::shutdown);
  m.def("set_device_count", &tnca::set_device_count);
  m.def("storage_address", &tnca::storage_address,
        "the device address of a tnca tensor's storage (its base, not data_ptr)");
}
