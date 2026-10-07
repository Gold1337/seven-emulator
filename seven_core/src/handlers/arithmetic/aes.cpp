// AES-NI (AESENC/AESENCLAST/AESDEC/AESDECLAST/AESIMC/AESKEYGENASSIST) and
// PCLMULQDQ: legacy SSE encodings plus their VEX 128/256 forms. CNG
// (bcryptprimitives) uses these internally, so Windows crypto call paths
// require them even for non-crypto guests.

#include <array>
#include <cstring>

#include "seven/handlers_fwd.hpp"
#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

using big_uint = seven::SimdUint;

constexpr std::size_t kXmmBytes = 16;
constexpr std::size_t kYmmBytes = 32;

bool aes_is_vector_register(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  return (value >= xmm0 && value < xmm0 + 32) || (value >= ymm0 && value < ymm0 + 32) || (value >= zmm0 && value < zmm0 + 32);
}

std::size_t aes_xmm_index(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= xmm0 && value < xmm0 + 32) return static_cast<std::size_t>(value - xmm0);
  if (value >= ymm0 && value < ymm0 + 32) return static_cast<std::size_t>(value - ymm0);
  if (value >= zmm0 && value < zmm0 + 32) return static_cast<std::size_t>(value - zmm0);
  return 0;
}

big_uint aes_mask(std::size_t bytes) {
  if (bytes >= sizeof(big_uint)) {
    big_uint all_ones(0);
    return ~all_ones;
  }
  return (big_uint(1) << (bytes * 8)) - 1;
}

// Reads a full-width (16 or 32 byte) operand into little-endian bytes. Bit i of
// a guest XMM register is byte i/8, bit i%8, matching the helpers above.
std::array<std::uint8_t, kYmmBytes> aes_read_operand_bytes(ExecutionContext& ctx, std::uint32_t index, std::size_t width, bool* ok) {
  std::array<std::uint8_t, kYmmBytes> out{};
  if (ctx.instr.op_kind(index) == iced_x86::OpKind::Register) {
    const auto reg = ctx.instr.op_register(index);
    if (!aes_is_vector_register(reg)) {
      if (ok != nullptr) *ok = false;
      return out;
    }
    const auto bits = ctx.state.vectors[aes_xmm_index(reg)].value;
    for (std::size_t i = 0; i < width; ++i) {
      out[i] = static_cast<std::uint8_t>((bits >> (i * 8)) & 0xFFu);
    }
    if (ok != nullptr) *ok = true;
    return out;
  }

  if (ctx.instr.op_kind(index) == iced_x86::OpKind::Memory) {
    const auto address = detail::memory_address(ctx);
    if (!ctx.memory.read(address, out.data(), width)) {
      if (ok != nullptr) *ok = false;
      return out;
    }
    if (ok != nullptr) *ok = true;
    return out;
  }

  if (ok != nullptr) *ok = false;
  return out;
}

void aes_write_register_bytes(ExecutionContext& ctx, std::size_t width, const std::uint8_t* bytes, bool zero_upper) {
  const auto dst_reg = ctx.instr.op_register(0);
  auto& slot = ctx.state.vectors[aes_xmm_index(dst_reg)].value;
  if (zero_upper) {
    slot = 0;
  }
  for (std::size_t i = 0; i < width; ++i) {
    const auto byte = static_cast<std::uint64_t>(bytes[i]);
    slot &= ~(static_cast<big_uint>(0xFF) << (i * 8));
    slot |= static_cast<big_uint>(byte) << (i * 8);
  }
}

// AES state bytes: byte index i of the XMM is state[i]; column c = bytes
// {4c..4c+3}, row r = bytes {r, r+4, r+8, r+12}.

constexpr std::array<std::uint8_t, 256> kAesSbox = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
};

constexpr std::array<std::uint8_t, 256> kAesInvSbox = {
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
};

std::uint8_t aes_xtime(std::uint8_t value) {
  return static_cast<std::uint8_t>(static_cast<std::uint8_t>(value << 1) ^ ((value & 0x80u) != 0 ? 0x1B : 0x00));
}

// Multiply in GF(2^8) with the AES reduction polynomial.
std::uint8_t aes_gmul(std::uint8_t a, std::uint8_t b) {
  std::uint8_t result = 0;
  auto x = a;
  auto y = b;
  while (y != 0) {
    if ((y & 1u) != 0) {
      result = static_cast<std::uint8_t>(result ^ x);
    }
    x = aes_xtime(x);
    y >>= 1;
  }
  return result;
}

std::uint32_t aes_sub_word(std::uint32_t value) {
  return (static_cast<std::uint32_t>(kAesSbox[(value >> 0) & 0xFFu]) << 0) |
         (static_cast<std::uint32_t>(kAesSbox[(value >> 8) & 0xFFu]) << 8) |
         (static_cast<std::uint32_t>(kAesSbox[(value >> 16) & 0xFFu]) << 16) |
         (static_cast<std::uint32_t>(kAesSbox[(value >> 24) & 0xFFu]) << 24);
}

std::uint32_t aes_rot_word(std::uint32_t value) {
  return (value << 24) | (value >> 8);
}

void aes_shift_rows(std::array<std::uint8_t, kXmmBytes>& state) {
  std::array<std::uint8_t, kXmmBytes> out{};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      out[r + 4 * c] = state[r + 4 * ((c + r) % 4)];
    }
  }
  state = out;
}

void aes_inv_shift_rows(std::array<std::uint8_t, kXmmBytes>& state) {
  std::array<std::uint8_t, kXmmBytes> out{};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      out[r + 4 * c] = state[r + 4 * ((c + 4 - (r % 4)) % 4)];
    }
  }
  state = out;
}

void aes_sub_bytes(std::array<std::uint8_t, kXmmBytes>& state) {
  for (auto& byte : state) {
    byte = kAesSbox[byte];
  }
}

void aes_inv_sub_bytes(std::array<std::uint8_t, kXmmBytes>& state) {
  for (auto& byte : state) {
    byte = kAesInvSbox[byte];
  }
}

void aes_mix_columns(std::array<std::uint8_t, kXmmBytes>& state) {
  for (std::size_t c = 0; c < 4; ++c) {
    const auto a0 = state[4 * c + 0];
    const auto a1 = state[4 * c + 1];
    const auto a2 = state[4 * c + 2];
    const auto a3 = state[4 * c + 3];
    state[4 * c + 0] = static_cast<std::uint8_t>(aes_gmul(a0, 2) ^ aes_gmul(a1, 3) ^ a2 ^ a3);
    state[4 * c + 1] = static_cast<std::uint8_t>(a0 ^ aes_gmul(a1, 2) ^ aes_gmul(a2, 3) ^ a3);
    state[4 * c + 2] = static_cast<std::uint8_t>(a0 ^ a1 ^ aes_gmul(a2, 2) ^ aes_gmul(a3, 3));
    state[4 * c + 3] = static_cast<std::uint8_t>(aes_gmul(a0, 3) ^ a1 ^ a2 ^ aes_gmul(a3, 2));
  }
}

void aes_inv_mix_columns(std::array<std::uint8_t, kXmmBytes>& state) {
  for (std::size_t c = 0; c < 4; ++c) {
    const auto a0 = state[4 * c + 0];
    const auto a1 = state[4 * c + 1];
    const auto a2 = state[4 * c + 2];
    const auto a3 = state[4 * c + 3];
    state[4 * c + 0] = static_cast<std::uint8_t>(aes_gmul(a0, 0x0E) ^ aes_gmul(a1, 0x0B) ^ aes_gmul(a2, 0x0D) ^ aes_gmul(a3, 0x09));
    state[4 * c + 1] = static_cast<std::uint8_t>(aes_gmul(a0, 0x09) ^ aes_gmul(a1, 0x0E) ^ aes_gmul(a2, 0x0B) ^ aes_gmul(a3, 0x0D));
    state[4 * c + 2] = static_cast<std::uint8_t>(aes_gmul(a0, 0x0D) ^ aes_gmul(a1, 0x09) ^ aes_gmul(a2, 0x0E) ^ aes_gmul(a3, 0x0B));
    state[4 * c + 3] = static_cast<std::uint8_t>(aes_gmul(a0, 0x0B) ^ aes_gmul(a1, 0x0D) ^ aes_gmul(a2, 0x09) ^ aes_gmul(a3, 0x0E));
  }
}

enum class aes_round_kind {
  enc,
  enc_last,
  dec,
  dec_last,
  imc,
};

std::array<std::uint8_t, kYmmBytes> aes_apply_round(const std::array<std::uint8_t, kYmmBytes>& dst,
                                                    const std::array<std::uint8_t, kYmmBytes>& src, std::size_t width,
                                                    aes_round_kind kind) {
  auto out = dst;
  for (std::size_t lane = 0; lane < width; lane += kXmmBytes) {
    std::array<std::uint8_t, kXmmBytes> state{};
    std::memcpy(state.data(), dst.data() + lane, kXmmBytes);

    switch (kind) {
    case aes_round_kind::enc:
      aes_shift_rows(state);
      aes_sub_bytes(state);
      aes_mix_columns(state);
      for (std::size_t i = 0; i < kXmmBytes; ++i) state[i] ^= src[lane + i];
      break;
    case aes_round_kind::enc_last:
      aes_shift_rows(state);
      aes_sub_bytes(state);
      for (std::size_t i = 0; i < kXmmBytes; ++i) state[i] ^= src[lane + i];
      break;
    case aes_round_kind::dec:
      aes_inv_shift_rows(state);
      aes_inv_sub_bytes(state);
      aes_inv_mix_columns(state);
      for (std::size_t i = 0; i < kXmmBytes; ++i) state[i] ^= src[lane + i];
      break;
    case aes_round_kind::dec_last:
      aes_inv_shift_rows(state);
      aes_inv_sub_bytes(state);
      for (std::size_t i = 0; i < kXmmBytes; ++i) state[i] ^= src[lane + i];
      break;
    case aes_round_kind::imc:
      // AESIMC ignores the (register-form) first operand: the result is
      // InvMixColumns of the source operand alone.
      std::memcpy(state.data(), src.data() + lane, kXmmBytes);
      aes_inv_mix_columns(state);
      break;
    }

    std::memcpy(out.data() + lane, state.data(), kXmmBytes);
  }
  return out;
}

ExecutionResult aes_round_handler(ExecutionContext& ctx, std::size_t width, aes_round_kind kind) {
  bool ok = false;
  const auto dst = aes_read_operand_bytes(ctx, 0, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src = aes_read_operand_bytes(ctx, 1, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));

  const auto result = aes_apply_round(dst, src, width, kind);
  aes_write_register_bytes(ctx, width, result.data(), ctx.instr.op_kind(0) == iced_x86::OpKind::Register &&
                                                     static_cast<std::uint32_t>(ctx.instr.op_register(0)) >=
                                                         static_cast<std::uint32_t>(iced_x86::Register::XMM0));
  return {};
}

ExecutionResult aes_round_dispatch(ExecutionContext& ctx, aes_round_kind kind) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register || !aes_is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }

  const auto dst_width = (static_cast<std::uint32_t>(ctx.instr.op_register(0)) >= static_cast<std::uint32_t>(iced_x86::Register::YMM0))
                             ? kYmmBytes
                             : kXmmBytes;
  return aes_round_handler(ctx, dst_width, kind);
}

}  // namespace

ExecutionResult handle_code_AESENC_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc);
}

ExecutionResult handle_code_AESENCLAST_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc_last);
}

ExecutionResult handle_code_AESDEC_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec);
}

ExecutionResult handle_code_AESDECLAST_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec_last);
}

ExecutionResult handle_code_AESIMC_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::imc);
}

ExecutionResult handle_code_VAESENC_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc);
}

ExecutionResult handle_code_VAESENCLAST_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc_last);
}

ExecutionResult handle_code_VAESDEC_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec);
}

ExecutionResult handle_code_VAESDECLAST_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec_last);
}

ExecutionResult handle_code_VAESIMC_XMM_XMMM128(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::imc);
}

ExecutionResult handle_code_VAESENC_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc);
}

ExecutionResult handle_code_VAESENCLAST_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::enc_last);
}

ExecutionResult handle_code_VAESDEC_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec);
}

ExecutionResult handle_code_VAESDECLAST_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return aes_round_dispatch(ctx, aes_round_kind::dec_last);
}

ExecutionResult handle_code_AESKEYGENASSIST_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register || !aes_is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }

  bool ok = false;
  auto src = aes_read_operand_bytes(ctx, 1, kXmmBytes, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));

  const auto rcon = static_cast<std::uint32_t>(ctx.instr.immediate8() & 0xFFu) << 24;
  std::array<std::uint8_t, kXmmBytes> out{};

  const auto apply = [&](std::size_t out_offset, std::size_t in_offset) {
    std::uint32_t word = 0;
    std::memcpy(&word, src.data() + in_offset, sizeof(word));
    const auto converted = aes_sub_word(aes_rot_word(word)) ^ rcon;
    std::memcpy(out.data() + out_offset, &converted, sizeof(converted));
  };

  apply(0, 0);                       // DEST[31:0] = SubWord(RotWord(SRC[31:0])) ^ RCON
  std::memcpy(out.data() + 4, src.data() + 4, 4);  // DEST[63:32] = SRC[63:32]
  apply(8, 8);                       // DEST[95:64] = SubWord(RotWord(SRC[95:64])) ^ RCON
  std::memcpy(out.data() + 12, src.data() + 12, 4);  // DEST[127:96] = SRC[127:96]

  aes_write_register_bytes(ctx, kXmmBytes, out.data(), false);
  return {};
}

ExecutionResult handle_code_VAESKEYGENASSIST_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  return handle_code_AESKEYGENASSIST_XMM_XMMM128_IMM8(ctx);
}

namespace {

// Carry-less multiplication of two 64-bit polynomials over GF(2).
std::array<std::uint8_t, kXmmBytes> pclmulqdq_64(std::uint64_t a, std::uint64_t b) {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
  for (int bit = 0; bit < 64; ++bit) {
    if (((b >> bit) & 1u) != 0) {
      const auto shifted = a << bit;
      lo ^= shifted;
      hi ^= (bit != 0) ? (a >> (64 - bit)) : 0;
    }
  }
  std::array<std::uint8_t, kXmmBytes> out{};
  std::memcpy(out.data(), &lo, sizeof(lo));
  std::memcpy(out.data() + 8, &hi, sizeof(hi));
  return out;
}

ExecutionResult pclmulqdq_handler(ExecutionContext& ctx, std::size_t width) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register || !aes_is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }

  bool ok = false;
  const auto dst = aes_read_operand_bytes(ctx, 0, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src = aes_read_operand_bytes(ctx, 1, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));

  const auto imm = ctx.instr.immediate8();
  const auto dst_sel = (imm & 0x01u) != 0 ? 8u : 0u;
  const auto src_sel = (imm & 0x10u) != 0 ? 8u : 0u;

  auto out = dst;
  for (std::size_t lane = 0; lane < width; lane += kXmmBytes) {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::memcpy(&a, dst.data() + lane + dst_sel, sizeof(a));
    std::memcpy(&b, src.data() + lane + src_sel, sizeof(b));
    const auto product = pclmulqdq_64(a, b);
    std::memcpy(out.data() + lane, product.data(), kXmmBytes);
  }

  aes_write_register_bytes(ctx, width, out.data(), false);
  return {};
}

}  // namespace

ExecutionResult handle_code_PCLMULQDQ_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  return pclmulqdq_handler(ctx, kXmmBytes);
}

ExecutionResult handle_code_VPCLMULQDQ_XMM_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  return pclmulqdq_handler(ctx, kXmmBytes);
}

ExecutionResult handle_code_VPCLMULQDQ_YMM_YMM_YMMM256_IMM8(ExecutionContext& ctx) {
  return pclmulqdq_handler(ctx, kYmmBytes);
}

}  // namespace seven::handlers
