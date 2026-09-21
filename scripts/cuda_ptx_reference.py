"""Small explicit CUDA Driver ABI helper and host-only cubin section reader."""
import ctypes as C
import hashlib
import pathlib
import struct


def elf_sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64 cubin")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if entry_size != 64 or count == 0 or names_index >= count:
        raise ValueError("Unexpected ELF section table")
    sections = []
    for index in range(count):
        start = offset + index * entry_size
        if start + 64 > len(data):
            raise ValueError("Truncated ELF section table")
        sections.append(struct.unpack_from("<IIQQQQIIQQ", data, start))
    strings_header = sections[names_index]
    strings = data[strings_header[4]:strings_header[4] + strings_header[5]]
    result = {}
    for header in sections:
        name_end = strings.find(b"\0", header[0])
        if name_end < 0:
            raise ValueError("Invalid ELF section name")
        name = strings[header[0]:name_end].decode("ascii")
        if header[1] == 8:  # SHT_NOBITS, such as dynamic shared memory.
            continue
        start, size = header[4], header[5]
        if start + size > len(data):
            raise ValueError("Truncated ELF section")
        if name in result:
            raise ValueError("Duplicate ELF section name")
        result[name] = data[start:start + size]
    return result


class DriverModule:
    """Load one cubin into Torch's current context; use kernelParams, not offsets."""
    def __init__(self, path, name, expected_sha256):
        self.driver = C.CDLL("libcuda.so.1")
        self.driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuModuleUnload.argtypes = [C.c_void_p]
        self.driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
        self.driver.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p,
                                            C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        for item in ["cuModuleLoadData", "cuModuleGetFunction", "cuModuleUnload", "cuCtxGetCurrent", "cuFuncGetAttribute", "cuLaunchKernel"]:
            getattr(self.driver, item).restype = C.c_int
        context = C.c_void_p()
        self.check(self.driver.cuCtxGetCurrent(C.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            raise RuntimeError("Torch must establish the CUDA context before module loading")
        self.module, self.function = C.c_void_p(), C.c_void_p()
        image_bytes = pathlib.Path(path).read_bytes()
        self.loaded_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if self.loaded_image_sha256 != expected_sha256:
            raise ValueError("Loaded cubin bytes differ from the checked identity")
        self.image = C.create_string_buffer(image_bytes)
        self.check(self.driver.cuModuleLoadData(C.byref(self.module), C.cast(self.image, C.c_void_p)), "cuModuleLoadData")
        try:
            self.check(self.driver.cuModuleGetFunction(C.byref(self.function), self.module, name.encode()), "cuModuleGetFunction")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"{operation} failed with CUDA result {code}")

    def attribute(self, key):
        result = C.c_int()
        self.check(self.driver.cuFuncGetAttribute(C.byref(result), key, self.function), "cuFuncGetAttribute")
        return result.value

    def launch_attention(self, pointers, stream, shared_bytes):
        if len(pointers) != 14:
            raise ValueError("Pinned original ABI has fourteen tensor pointers")
        # The saved PTX entry has fourteen pointers, five u32 sizes, followed
        # by the two zero-sized global/profile scratch pointers added by Triton.
        values = [C.c_uint64(value.data_ptr()) for value in pointers]
        values += [C.c_uint32(value) for value in [144, 16384, 2, 2, 2]]
        values += [C.c_uint64(0), C.c_uint64(0)]
        self._launch(values, [2, 1, 16], [128, 1, 1], stream, shared_bytes)

    def launch_natural_lse(self, source, output, stream):
        # The preserved pointwise PTX has XBLOCK=128, an i64 ks0 and i32
        # xnumel, plus its two zero-sized scratch pointers. No JIT is needed.
        values = [C.c_uint64(source.data_ptr()), C.c_uint64(output.data_ptr()),
                  C.c_uint64(144), C.c_uint32(2304), C.c_uint64(0), C.c_uint64(0)]
        self._launch(values, [18, 1, 1], [128, 1, 1], stream, 0)

    def _launch(self, values, grid, block, stream, shared_bytes):
        arguments = (C.c_void_p * len(values))(*(C.cast(C.byref(value), C.c_void_p) for value in values))
        self.check(self.driver.cuLaunchKernel(self.function, *grid, *block,
                                             shared_bytes, C.c_void_p(stream), arguments, None), "cuLaunchKernel")

    def close(self):
        if self.module.value:
            self.check(self.driver.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = C.c_void_p()
