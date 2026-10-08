#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

namespace {
using big_uint = seven::SimdUint;

big_uint mask(std::size_t width) {
  return (big_uint(1) << (width * 8)) - 1;
}


std::size_t xmm_index(iced_x86::Register reg) {
  return static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::XMM0));
}

std::size_t vector_index(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= xmm0 && value < xmm0 + 32) return static_cast<std::size_t>(value - xmm0);
  if (value >= ymm0 && value < ymm0 + 32) return static_cast<std::size_t>(value - ymm0);
  if (value >= zmm0 && value < zmm0 + 32) return static_cast<std::size_t>(value - zmm0);
  return 0;
}

ExecutionResult validate_memory_span(ExecutionContext& ctx, std::uint64_t base, std::size_t size, seven::MemoryAccessKind kind) {
  std::size_t offset = 0;
  while (offset < size) {
    const auto address = base + offset;
    const auto page_offset = static_cast<std::size_t>(address % seven::Memory::kPageSize);
    const auto chunk = std::min(size - offset, seven::Memory::kPageSize - page_offset);
    bool allowed = false;
    if (kind == seven::MemoryAccessKind::data_write) {
      allowed = ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
    } else {
      allowed = ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read)) ||
                ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write)) ||
                ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::execute));
    }
    if (!ctx.memory.is_mapped(address, chunk) || !allowed) {
      return detail::memory_fault(ctx, address);
    }
    offset += chunk;
  }
  return {};
}


std::uint64_t read_mmx(CpuState& state, iced_x86::Register reg);

ExecutionResult pmovmskb(ExecutionContext& ctx, std::size_t src_bytes, bool use_mmx, std::size_t dst_bytes) {
  SimdUint src = 0;
  if (use_mmx) {
    src = SimdUint(read_mmx(ctx.state, ctx.instr.op_register(1)));
  } else {
    const auto src_reg = ctx.instr.op_register(1);
    src = ctx.state.vectors[vector_index(src_reg)].value;
  }

  std::uint64_t mask = 0;
  for (std::size_t i = 0; i < src_bytes; ++i) {
    const std::uint64_t msb = ((src >> ((i * 8) + 7)) & SimdUint(1)) != SimdUint(0) ? 1u : 0u;
    mask |= (msb << i);
  }

  if (!detail::write_operand(ctx, 0, mask, dst_bytes)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult movmskpd(ExecutionContext& ctx, std::size_t dst_bytes) {
  const auto src_reg = ctx.instr.op_register(1);
  const auto src = ctx.state.vectors[vector_index(src_reg)].value;
  const auto mask = static_cast<std::uint64_t>(((src >> 63) & 1u) | (((src >> 127) & 1u) << 1));
  if (!detail::write_operand(ctx, 0, mask, dst_bytes)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

std::uint64_t read_mmx(CpuState& state, iced_x86::Register reg) {
  return state.mmx_get(static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::MM0)));
}

ExecutionResult store_masked_bytes(ExecutionContext& ctx, std::uint32_t data_index, std::uint32_t mask_index, std::size_t width, bool use_xmm) {
  const auto dest = detail::read_register(ctx.state, iced_x86::Register::RDI);
  std::array<std::uint8_t, 64> bytes{};
  std::array<std::uint8_t, 64> mask{};

  if (use_xmm) {
    const auto data_reg = ctx.instr.op_register(data_index);
    const auto mask_reg = ctx.instr.op_register(mask_index);
    const auto data = ctx.state.vectors[xmm_index(data_reg)].value;
    const auto mask_value = ctx.state.vectors[xmm_index(mask_reg)].value;
    for (std::size_t i = 0; i < width; ++i) {
      bytes[i] = static_cast<std::uint8_t>((data >> (8 * i)) & 0xFFu);
      mask[i] = static_cast<std::uint8_t>((mask_value >> (8 * i)) & 0xFFu);
    }
  } else {
    const auto data = read_mmx(ctx.state, ctx.instr.op_register(data_index));
    const auto mask_value = read_mmx(ctx.state, ctx.instr.op_register(mask_index));
    for (std::size_t i = 0; i < width; ++i) {
      bytes[i] = static_cast<std::uint8_t>((data >> (8 * i)) & 0xFFu);
      mask[i] = static_cast<std::uint8_t>((mask_value >> (8 * i)) & 0xFFu);
    }
  }

  for (std::size_t i = 0; i < width; ++i) {
    if ((mask[i] & 0x80u) != 0) {
      if (!ctx.memory.write(dest + i, &bytes[i], 1)) {
        return detail::memory_fault(ctx, dest + i);
      }
    }
  }
  return {};
}

ExecutionResult mmx_move(ExecutionContext& ctx, std::uint32_t dst_index, std::uint32_t src_index, std::size_t width) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, src_index, width, &ok);
  if (!ok) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!detail::write_operand(ctx, dst_index, value, width)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

void emms(CpuState& state) {
  state.x87_tags.fill(0x3);
}

uint8_t x87_ftw(const CpuState& state) {
  uint8_t ftw = 0;
  for (std::size_t phys = 0; phys < 8; ++phys) {
    if (state.x87_tags[phys] != 0x3) {
      ftw |= static_cast<uint8_t>(1u << phys);
    }
  }
  return ftw;
}

void store_x87_env(ExecutionContext& ctx, std::uint64_t base) {
  const std::uint16_t fcw = ctx.state.get_x87_control_word();
  const std::uint16_t fsw = ctx.state.get_x87_status_word();
  const std::uint16_t ftw = static_cast<std::uint16_t>(x87_ftw(ctx.state));
  const std::uint16_t zero16 = 0;
  const std::uint64_t zero64 = 0;
  ctx.memory.write(base + 0, &fcw, 2);
  ctx.memory.write(base + 2, &fsw, 2);
  ctx.memory.write(base + 4, &ftw, 2);
  ctx.memory.write(base + 6, &zero16, 2);
  ctx.memory.write(base + 8, &zero64, 8);
  ctx.memory.write(base + 16, &zero64, 8);
  ctx.memory.write(base + 24, &zero16, 2);
}

ExecutionResult load_x87_env(ExecutionContext& ctx, std::uint64_t base) {
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint16_t ftw = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 2)) return detail::memory_fault(ctx, base + 4);
  if ((fcw & 0xE0C0u) != 0) return detail::memory_fault(ctx, base);
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  for (std::size_t i = 0; i < 8; ++i) {
    ctx.state.x87_tags[ctx.state.x87_phys_index(i)] = ((ftw >> i) & 1u) ? 0x0 : 0x3;
  }
  return {};
}

void write_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys, X87Scalar value) {
  std::array<std::uint8_t, 16> raw{};
  x87_encoding::encode_ext80(value, raw.data());
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    ctx.memory.write(slot + i, &raw[i], 1);
  }
}

X87Scalar read_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys) {
  std::array<std::uint8_t, 16> raw{};
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    ctx.memory.read(slot + i, &raw[i], 1);
  }
  return x87_encoding::decode_ext80(raw.data());
}

void write_fpu_state(ExecutionContext& ctx, std::uint64_t base, std::size_t offset) {
  const std::uint16_t fcw = ctx.state.get_x87_control_word();
  const std::uint16_t fsw = ctx.state.get_x87_status_word();
  const std::uint8_t ftw = x87_ftw(ctx.state);
  const std::uint64_t zero64 = 0;
  ctx.memory.write(base + offset + 0, &fcw, 2);
  ctx.memory.write(base + offset + 2, &fsw, 2);
  ctx.memory.write(base + offset + 4, &ftw, 1);
  ctx.memory.write(base + offset + 5, &zero64, 1);
  ctx.memory.write(base + offset + 6, &zero64, 2);
  ctx.memory.write(base + offset + 8, &zero64, 8);
  ctx.memory.write(base + offset + 16, &zero64, 8);
}

ExecutionResult fsave(ExecutionContext& ctx, std::size_t env_size) {
  (void)env_size;
  const auto base = detail::memory_address(ctx);
  if ((base & 0x7) != 0) {
    return detail::memory_fault(ctx, base);
  }
  write_fpu_state(ctx, base, 0);
  for (std::size_t i = 0; i < 8; ++i) {
    write_fxsave_st(ctx, base, i, ctx.state.x87_get(i));
  }
  ctx.state.x87_reset();
  return {};
}

ExecutionResult frstor(ExecutionContext& ctx, std::size_t env_size) {
  (void)env_size;
  const auto base = detail::memory_address(ctx);
  if ((base & 0x7) != 0) {
    return detail::memory_fault(ctx, base);
  }
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint8_t ftw = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  for (std::size_t i = 0; i < 8; ++i) {
    if ((ftw >> i) & 1u) {
      ctx.state.x87_set(i, read_fxsave_st(ctx, base, i));
    } else {
      ctx.state.x87_mark_empty(i);
    }
  }
  return {};
}

ExecutionResult fxsave(ExecutionContext& ctx, bool /*is64*/) {
  const auto base = detail::memory_address(ctx);
  if ((base & 0xF) != 0) {
    return detail::memory_fault(ctx, base);
  }
  if (const auto span = validate_memory_span(ctx, base, 0x200, seven::MemoryAccessKind::data_write); !span.ok()) return span;
  if (!ctx.memory.write(base + 0, &ctx.state.x87_control_word, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.write(base + 2, &ctx.state.x87_status_word, 2)) return detail::memory_fault(ctx, base + 2);
  const auto ftw = x87_ftw(ctx.state);
  if (!ctx.memory.write(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  const std::uint8_t zero8 = 0;
  if (!ctx.memory.write(base + 5, &zero8, 1)) return detail::memory_fault(ctx, base + 5);
  const std::uint16_t fop = 0;
  if (!ctx.memory.write(base + 6, &fop, 2)) return detail::memory_fault(ctx, base + 6);
  const std::uint64_t zero64 = 0;
  if (!ctx.memory.write(base + 8, &zero64, 8)) return detail::memory_fault(ctx, base + 8);
  if (!ctx.memory.write(base + 16, &zero64, 8)) return detail::memory_fault(ctx, base + 16);
  if (!ctx.memory.write(base + 24, &ctx.state.mxcsr, 4)) return detail::memory_fault(ctx, base + 24);
  const std::uint32_t mxcsr_mask = 0xFFFFu;
  if (!ctx.memory.write(base + 28, &mxcsr_mask, 4)) return detail::memory_fault(ctx, base + 28);

  for (std::size_t phys = 0; phys < 8; ++phys) {
    write_fxsave_st(ctx, base, phys, ctx.state.x87_stack[phys]);
  }
  for (std::size_t i = 0; i < 16; ++i) {
    const auto value = ctx.state.vectors[i].value;
    std::array<std::uint8_t, 16> raw{};
    for (std::size_t b = 0; b < raw.size(); ++b) {
      raw[b] = static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFu);
    }
    const auto slot = base + 160 + (i * 16);
    for (std::size_t b = 0; b < raw.size(); ++b) {
      if (!ctx.memory.write(slot + b, &raw[b], 1)) return detail::memory_fault(ctx, slot + b);
    }
  }
  for (std::size_t offset = 416; offset < 512; ++offset) {
    if (!ctx.memory.write(base + offset, &zero8, 1)) return detail::memory_fault(ctx, base + offset);
  }
  return {};
}

ExecutionResult fxrstor(ExecutionContext& ctx, bool /*is64*/) {
  const auto base = detail::memory_address(ctx);
  if ((base & 0xF) != 0) {
    return detail::memory_fault(ctx, base);
  }
  if (const auto span = validate_memory_span(ctx, base, 0x200, seven::MemoryAccessKind::data_read); !span.ok()) return span;
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint8_t ftw = 0;
  std::uint32_t mxcsr = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  if (!ctx.memory.read(base + 24, &mxcsr, 4)) return detail::memory_fault(ctx, base + 24);
  fcw &= static_cast<std::uint16_t>(~0xE0C0u);
  mxcsr &= 0x0000FFFFu;
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  ctx.state.mxcsr = mxcsr;
  for (std::size_t phys = 0; phys < 8; ++phys) {
    if ((ftw >> phys) & 1u) {
      ctx.state.x87_stack[phys] = read_fxsave_st(ctx, base, phys);
      ctx.state.x87_tags[phys] = (ctx.state.x87_stack[phys] == 0) ? 0x1 : 0x0;
    } else {
      ctx.state.x87_tags[phys] = 0x3;
    }
  }
  for (std::size_t i = 0; i < 16; ++i) {
    std::array<std::uint8_t, 16> raw{};
    const auto slot = base + 160 + (i * 16);
    for (std::size_t b = 0; b < raw.size(); ++b) {
      if (!ctx.memory.read(slot + b, &raw[b], 1)) return detail::memory_fault(ctx, slot + b);
    }
    seven::SimdUint value = 0;
    for (std::size_t b = 0; b < raw.size(); ++b) {
      value |= (seven::SimdUint(raw[b]) << (8 * b));
    }
    ctx.state.vectors[i].value = value;
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_MOVD_MM_RM32(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 4); }
ExecutionResult handle_code_MOVQ_MM_RM64(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVD_RM32_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 4); }
ExecutionResult handle_code_MOVQ_RM64_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVQ_MM_MMM64(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVQ_MMM64_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }

ExecutionResult handle_code_FNINIT(ExecutionContext& ctx) {
  ctx.state.x87_reset();
  return {};
}

ExecutionResult handle_code_FINIT(ExecutionContext& ctx) {
  ctx.state.x87_reset();
  return {};
}

ExecutionResult handle_code_EMMS(ExecutionContext& ctx) {
  emms(ctx.state);
  return {};
}

ExecutionResult handle_code_FEMMS(ExecutionContext& ctx) {
  emms(ctx.state);
  return {};
}

ExecutionResult handle_code_FNCLEX(ExecutionContext& ctx) {
  ctx.state.x87_status_word &= ~std::uint16_t(0xBFu);
  return {};
}

ExecutionResult handle_code_FLD1(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(1)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2T(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(3.32192809488736234787)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2E(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(1.44269504088896340736)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDPI(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(3.14159265358979323846)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLG2(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0.30102999566398119521)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLN2(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0.69314718055994530942)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDZ(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FCHS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return -v; });
}

ExecutionResult handle_code_FABS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return seven::abs(v); });
}

ExecutionResult handle_code_FSQRT(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  if (value < 0) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = seven::sqrt(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  if (auto r = x87_precision_if_changed(ctx, value, result); !r.ok()) return r;
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FRNDINT(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar rounded = x87_round_to_control(ctx.state, value);
  if (rounded != value) {
    auto result = x87_exception(ctx, kX87ExceptionPrecision);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, rounded);
  return {};
}

ExecutionResult handle_code_FSIN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = seven::sin(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FCOS(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = seven::cos(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FSINCOS(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar x = ctx.state.x87_get(0);
  const X87Scalar cosine = seven::cos(x);
  const X87Scalar sine = seven::sin(x);
  if (x != 0 && (cosine == 0 || sine == 0)) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(cosine, x, 0) | x87_classify_result(sine, x, 0)); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, cosine);
  if (!ctx.state.x87_push(sine)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FPTAN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = seven::tan(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  if (!ctx.state.x87_push(1)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FPATAN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  ctx.state.x87_set(1, seven::atan2(y, x));
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_F2XM1(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  if (value < -1 || value > 1) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = seven::pow(X87Scalar(2), value) - X87Scalar(1);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FYL2X(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  if (x <= 0) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = y * seven::log2(x);
  if (y != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, y, x); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_FYL2XP1(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  if (x <= -1) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = y * seven::log2(x + 1);
  if (y != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, y, x + 1); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_FSCALE(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar a = ctx.state.x87_get(0);
  const X87Scalar b = ctx.state.x87_get(1);
  const X87Scalar result = seven::ldexp(a, static_cast<int>(seven::trunc(b)));
  if (a != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, a, b); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FLDENV_M14BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_FLDENV_M28BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_FLD_M32FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 4);
}

ExecutionResult handle_code_FLD_M64FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 8);
}

ExecutionResult handle_code_FLD_M80FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 10);
}

ExecutionResult handle_code_FLD_STI(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register || ctx.instr.op_kind(1) != iced_x86::OpKind::Register) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!ctx.state.x87_push(ctx.state.x87_get(x87_st_index(ctx.instr.op_register(0))))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FNSTENV_M14BYTE(ExecutionContext& ctx) {
  store_x87_env(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult handle_code_FSTENV_M14BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M14BYTE(ctx);
}

ExecutionResult handle_code_FNSTENV_M28BYTE(ExecutionContext& ctx) {
  store_x87_env(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult handle_code_FSTENV_M28BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M28BYTE(ctx);
}

ExecutionResult handle_code_FLDCW_M2BYTE(ExecutionContext& ctx) {
  std::uint16_t value = 0;
  if (!detail::read_operand_checked(ctx, 0, 2, value).ok()) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  // Real x86 accepts any 16-bit control word: FLDCW never faults on the VALUE.
  // The previous "reserved bits" check (value & 0xE0C0: PM mask + reserved bits
  // 6-7 / 13-15) rejected every legitimate control word - 0x027F (the Windows
  // initial CW), 0x037F (FINIT) and the guest-computed 0x0E7F (CW | 0x0C00,
  // round-toward-zero) all have bit 5/6 set - so any guest that switched
  // rounding mode via fldcw (the canonical truncate pattern CW|0x0C00) took a
  // spurious page fault. QEMU/unicorn store the value raw; do the same.
  ctx.state.set_x87_control_word(value);
  return {};
}

ExecutionResult handle_code_FNSTCW_M2BYTE(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.get_x87_control_word(), 2)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FST_M32FP(ExecutionContext& ctx) {
  return x87_store_to_memory(ctx, 4, false);
}

ExecutionResult handle_code_FST_M64FP(ExecutionContext& ctx) {
  return x87_store_to_memory(ctx, 8, false);
}

ExecutionResult handle_code_FST_STI(ExecutionContext& ctx) {
  // `fst st(i)` (DD D0+i) decodes with a SINGLE register operand (op0 = ST(i));
  // the stored source ST(0) is implicit. x87_reg_move's op1-based source would
  // read a nonexistent operand slot (see x87_store_st0_to_sti for details).
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto idx = x87_st_index(ctx.instr.op_register(0));
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  ctx.state.x87_set(idx, ctx.state.x87_get(0));
  return {};
}

ExecutionResult handle_code_FNSTSW_M2BYTE(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.get_x87_status_word(), 2)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FSTSW_M2BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTSW_M2BYTE(ctx);
}

ExecutionResult handle_code_FNSTSW_AX(ExecutionContext& ctx) {
  detail::write_register(ctx.state, iced_x86::Register::AX, ctx.state.get_x87_status_word(), 2);
  return {};
}

ExecutionResult handle_code_FSTSW_AX(ExecutionContext& ctx) {
  return handle_code_FNSTSW_AX(ctx);
}

ExecutionResult handle_code_FXSAVE_M512BYTE(ExecutionContext& ctx) {
  return fxsave(ctx, false);
}

ExecutionResult handle_code_FXSAVE64_M512BYTE(ExecutionContext& ctx) {
  return fxsave(ctx, true);
}

ExecutionResult handle_code_FXRSTOR_M512BYTE(ExecutionContext& ctx) {
  return fxrstor(ctx, false);
}

ExecutionResult handle_code_FXRSTOR64_M512BYTE(ExecutionContext& ctx) {
  return fxrstor(ctx, true);
}

ExecutionResult handle_code_FSAVE_M94BYTE(ExecutionContext& ctx) {
  return fsave(ctx, 94);
}

ExecutionResult handle_code_FSAVE_M108BYTE(ExecutionContext& ctx) {
  return fsave(ctx, 108);
}

ExecutionResult handle_code_FRSTOR_M94BYTE(ExecutionContext& ctx) {
  return frstor(ctx, 94);
}

ExecutionResult handle_code_FRSTOR_M108BYTE(ExecutionContext& ctx) {
  return frstor(ctx, 108);
}

ExecutionResult handle_code_FILD_M16INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 2);
}

ExecutionResult handle_code_FILD_M32INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 4);
}

ExecutionResult handle_code_FILD_M64INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 8);
}

ExecutionResult handle_code_FIST_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, false, false);
}

ExecutionResult handle_code_FIST_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, false, false);
}

ExecutionResult handle_code_FIST_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, false, false);
}

ExecutionResult handle_code_FISTP_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, true, false);
}

ExecutionResult handle_code_FISTP_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, true, false);
}

ExecutionResult handle_code_FISTP_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, true, false);
}

ExecutionResult handle_code_FISTTP_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, true, true);
}

ExecutionResult handle_code_FISTTP_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, true, true);
}

ExecutionResult handle_code_FISTTP_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, true, true);
}

ExecutionResult handle_code_MASKMOVQ_R_DI_MM_MM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 8, false);
}

ExecutionResult handle_code_MASKMOVDQU_R_DI_XMM_XMM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 16, true);
}

ExecutionResult handle_code_VEX_VMASKMOVDQU_R_DI_XMM_XMM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 16, true);
}

ExecutionResult handle_code_PMOVMSKB_R32_MM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 8, true, 4);
}

ExecutionResult handle_code_PMOVMSKB_R64_MM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 8, true, 8);
}

ExecutionResult handle_code_PMOVMSKB_R32_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 4);
}

ExecutionResult handle_code_PMOVMSKB_R64_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 8);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R32_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 4);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R64_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 8);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R32_YMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 32, false, 4);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R64_YMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 32, false, 8);
}
ExecutionResult handle_code_MOVMSKPD_R32_XMM(ExecutionContext& ctx) {
  return movmskpd(ctx, 4);
}
ExecutionResult handle_code_MOVMSKPD_R64_XMM(ExecutionContext& ctx) {
  return movmskpd(ctx, 8);
}
ExecutionResult handle_code_MOVQ2DQ_XMM_MM(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src = read_mmx(ctx.state, ctx.instr.op_register(1));
  auto& slot = ctx.state.vectors[vector_index(dst_reg)].value;
  slot = (slot & ~mask(16)) | big_uint(src);
  return {};
}
ExecutionResult handle_code_MOVDQ2Q_MM_XMM(ExecutionContext& ctx) {
  const auto src_reg = ctx.instr.op_register(1);
  const auto value = static_cast<std::uint64_t>(ctx.state.vectors[vector_index(src_reg)].value & mask(8));
  ctx.state.mmx_set(static_cast<std::size_t>(static_cast<std::uint32_t>(ctx.instr.op_register(0)) - static_cast<std::uint32_t>(iced_x86::Register::MM0)), value);
  return {};
}

// ── MMX packed integer arithmetic ────────────────────────────────────────────
// The MMX register-file forms (0F 6x-7x / 0F D4-F8 opcodes with mod=11) had no
// handlers: any guest using MMX packed instructions (as produced by MSVC for
// intrinsics like _mm_add_pi16) stopped with unsupported_instruction. The SSE
// XMM forms in simd_int.cpp/simd_pack.cpp do not cover these because iced
// decodes MMX and SSE forms as distinct Code values. All handlers below read
// the source through detail::read_operand (which routes MM registers to the
// MMX file and memory operands through the MMU) and write through
// detail::write_operand (which lands in mmx_set, marking the aliased x87 tag
// valid, matching the real aliasing behaviour of MMX over the FPU stack).

namespace {

std::uint64_t mm_read_source(ExecutionContext& ctx, std::uint32_t op_index, bool& ok) {
  return detail::read_operand(ctx, op_index, 8, &ok);
}

template <typename T>
inline T mm_lane_load(std::uint64_t value, std::size_t byte_offset) {
  T out{};
  std::memcpy(&out, reinterpret_cast<const std::uint8_t*>(&value) + byte_offset, sizeof(T));
  return out;
}

template <typename T>
inline void mm_lane_store(std::uint64_t& value, std::size_t byte_offset, T lane) {
  std::memcpy(reinterpret_cast<std::uint8_t*>(&value) + byte_offset, &lane, sizeof(T));
}

template <typename T, typename Fn>
ExecutionResult mm_binary(ExecutionContext& ctx, Fn&& fn) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t rhs = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = lhs;
  for (std::size_t lane = 0; lane < 8; lane += sizeof(T)) {
    mm_lane_store(out, lane, fn(mm_lane_load<T>(lhs, lane), mm_lane_load<T>(rhs, lane)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

template <typename T>
inline T mm_sat_add_unsigned(T a, T b) {
  const auto wide = static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b);
  constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<T>::max());
  return static_cast<T>(wide > max ? max : wide);
}

template <typename T>
inline T mm_sat_sub_unsigned(T a, T b) {
  return a > b ? static_cast<T>(a - b) : T{0};
}

template <typename T>
inline T mm_sat_add_signed(T a, T b) {
  using W = std::make_unsigned_t<T>;
  const auto sum = static_cast<T>(static_cast<W>(a) + static_cast<W>(b));
  if (a < 0 && b < 0 && sum >= 0) return std::numeric_limits<T>::min();
  if (a >= 0 && b >= 0 && sum < 0) return std::numeric_limits<T>::max();
  return sum;
}

template <typename T>
inline T mm_sat_sub_signed(T a, T b) {
  using W = std::make_unsigned_t<T>;
  const auto diff = static_cast<T>(static_cast<W>(a) - static_cast<W>(b));
  if (a >= 0 && b < 0 && diff < 0) return std::numeric_limits<T>::max();
  if (a < 0 && b >= 0 && diff >= 0) return std::numeric_limits<T>::min();
  return diff;
}

inline std::uint64_t mm_bitwise(ExecutionContext& ctx, std::uint64_t (Op)(std::uint64_t, std::uint64_t), bool& ok) {
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return 0;
  return Op(lhs, mm_read_source(ctx, 1, ok));
}

ExecutionResult mm_bitwise_finish(ExecutionContext& ctx, std::uint64_t value) {
  return detail::write_operand(ctx, 0, value, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

// Shift helpers: MMX counts come from imm8 or the low 64 bits of the source and
// are masked to 6 bits; per-element counts at or above the element width force
// the defined zero/sign-fill results.
template <typename T>
inline T mm_shift_left(T value, std::uint32_t count) {
  if (count >= sizeof(T) * 8) return T{0};
  return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(value) << count);
}

template <typename T>
inline T mm_shift_right_logical(T value, std::uint32_t count) {
  if (count >= sizeof(T) * 8) return T{0};
  return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(value) >> count);
}

template <typename T>
inline T mm_shift_right_arithmetic(T value, std::uint32_t count) {
  if (count >= sizeof(T) * 8) return value < 0 ? static_cast<T>(-1) : T{0};
  return static_cast<T>(value >> count);
}

std::uint64_t mm_read_source_or_zero(ExecutionContext& ctx, std::uint32_t op_index) {
  bool ok = false;
  return mm_read_source(ctx, op_index, ok);
}

}  // namespace

#define KUBERA_MMX_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return mm_binary<type>(ctx, fn); }
#define KUBERA_MMX_BITWISE(name, op) ExecutionResult handle_code_##name(ExecutionContext& ctx) { \
  bool ok = true; \
  const auto value = mm_bitwise(ctx, op, ok); \
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx)); \
  return mm_bitwise_finish(ctx, value); \
}

KUBERA_MMX_BIN(PADDB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); })
KUBERA_MMX_BIN(PADDW_MM_MMM64, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a + b); })
KUBERA_MMX_BIN(PADDD_MM_MMM64, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_MMX_BIN(PADDQ_MM_MMM64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_MMX_BIN(PSUBB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a - b); })
KUBERA_MMX_BIN(PSUBW_MM_MMM64, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a - b); })
KUBERA_MMX_BIN(PSUBD_MM_MMM64, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_MMX_BIN(PSUBQ_MM_MMM64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_MMX_BIN(PADDUSB_MM_MMM64, std::uint8_t, mm_sat_add_unsigned<std::uint8_t>)
KUBERA_MMX_BIN(PADDUSW_MM_MMM64, std::uint16_t, mm_sat_add_unsigned<std::uint16_t>)
KUBERA_MMX_BIN(PADDSB_MM_MMM64, std::int8_t, mm_sat_add_signed<std::int8_t>)
KUBERA_MMX_BIN(PADDSW_MM_MMM64, std::int16_t, mm_sat_add_signed<std::int16_t>)
KUBERA_MMX_BIN(PSUBUSB_MM_MMM64, std::uint8_t, mm_sat_sub_unsigned<std::uint8_t>)
KUBERA_MMX_BIN(PSUBUSW_MM_MMM64, std::uint16_t, mm_sat_sub_unsigned<std::uint16_t>)
KUBERA_MMX_BIN(PSUBSB_MM_MMM64, std::int8_t, mm_sat_sub_signed<std::int8_t>)
KUBERA_MMX_BIN(PSUBSW_MM_MMM64, std::int16_t, mm_sat_sub_signed<std::int16_t>)
KUBERA_MMX_BIN(PCMPEQB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return a == b ? static_cast<std::uint8_t>(0xFF) : std::uint8_t{0}; })
KUBERA_MMX_BIN(PCMPEQW_MM_MMM64, std::uint16_t, [](auto a, auto b) { return a == b ? static_cast<std::uint16_t>(0xFFFF) : std::uint16_t{0}; })
KUBERA_MMX_BIN(PCMPEQD_MM_MMM64, std::uint32_t, [](auto a, auto b) { return a == b ? static_cast<std::uint32_t>(0xFFFFFFFFu) : std::uint32_t{0}; })
KUBERA_MMX_BIN(PCMPGTB_MM_MMM64, std::int8_t, [](auto a, auto b) { return a > b ? static_cast<std::uint8_t>(0xFF) : std::uint8_t{0}; })
KUBERA_MMX_BIN(PCMPGTW_MM_MMM64, std::int16_t, [](auto a, auto b) { return a > b ? static_cast<std::uint16_t>(0xFFFF) : std::uint16_t{0}; })
KUBERA_MMX_BIN(PCMPGTD_MM_MMM64, std::int32_t, [](auto a, auto b) { return a > b ? static_cast<std::uint32_t>(0xFFFFFFFFu) : std::uint32_t{0}; })
KUBERA_MMX_BIN(PAVGB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>((static_cast<std::uint32_t>(a) + b + 1) >> 1); })
KUBERA_MMX_BIN(PAVGW_MM_MMM64, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>((static_cast<std::uint32_t>(a) + b + 1) >> 1); })
KUBERA_MMX_BIN(PMINUB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return a < b ? a : b; })
KUBERA_MMX_BIN(PMAXUB_MM_MMM64, std::uint8_t, [](auto a, auto b) { return a > b ? a : b; })
KUBERA_MMX_BIN(PMINSW_MM_MMM64, std::int16_t, [](auto a, auto b) { return a < b ? a : b; })
KUBERA_MMX_BIN(PMAXSW_MM_MMM64, std::int16_t, [](auto a, auto b) { return a > b ? a : b; })
KUBERA_MMX_BIN(PMULLW_MM_MMM64, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>(static_cast<std::uint32_t>(static_cast<std::uint16_t>(a)) * static_cast<std::uint16_t>(b));
})
KUBERA_MMX_BIN(PMULHW_MM_MMM64, std::int16_t, [](auto a, auto b) {
  return static_cast<std::int16_t>((static_cast<std::int32_t>(static_cast<std::int16_t>(a)) * static_cast<std::int32_t>(static_cast<std::int16_t>(b))) >> 16);
})
KUBERA_MMX_BIN(PMULHUW_MM_MMM64, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>((static_cast<std::uint32_t>(a) * static_cast<std::uint32_t>(b)) >> 16);
})
KUBERA_MMX_BITWISE(PAND_MM_MMM64, [](std::uint64_t a, std::uint64_t b) { return a & b; })
KUBERA_MMX_BITWISE(POR_MM_MMM64, [](std::uint64_t a, std::uint64_t b) { return a | b; })
KUBERA_MMX_BITWISE(PXOR_MM_MMM64, [](std::uint64_t a, std::uint64_t b) { return a ^ b; })
KUBERA_MMX_BITWISE(PANDN_MM_MMM64, [](std::uint64_t a, std::uint64_t b) { return (~a) & b; })

ExecutionResult handle_code_PMULUDQ_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t rhs = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t out = static_cast<std::uint64_t>(static_cast<std::uint32_t>(lhs)) * static_cast<std::uint64_t>(static_cast<std::uint32_t>(rhs));
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PMADDWD_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t rhs = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = 0;
  for (std::size_t pair = 0; pair < 2; ++pair) {
    const std::size_t off = pair * 4;
    const std::int32_t a0 = mm_lane_load<std::int16_t>(lhs, off);
    const std::int32_t a1 = mm_lane_load<std::int16_t>(lhs, off + 2);
    const std::int32_t b0 = mm_lane_load<std::int16_t>(rhs, off);
    const std::int32_t b1 = mm_lane_load<std::int16_t>(rhs, off + 2);
    const std::int64_t sum = static_cast<std::int64_t>(a0 * b0) + static_cast<std::int64_t>(a1 * b1);
    const std::int32_t saturated = sum > INT32_MAX ? INT32_MAX : (sum < INT32_MIN ? INT32_MIN : static_cast<std::int32_t>(sum));
    mm_lane_store(out, off, saturated);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PSADBW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t rhs = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    const std::uint8_t a = mm_lane_load<std::uint8_t>(lhs, i);
    const std::uint8_t b = mm_lane_load<std::uint8_t>(rhs, i);
    sum += a > b ? a - b : b - a;
  }
  const std::uint64_t out = static_cast<std::uint16_t>(sum);
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

namespace {

template <typename T>
ExecutionResult mm_shift(ExecutionContext& ctx, T (*fn)(T, std::uint32_t)) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const std::uint64_t lhs = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  // Imm forms decode with a single operand (dst); reg/memory forms carry the
  // count in op1. The x86 count semantics mask to 6 bits either way.
  std::uint64_t count = 0;
  if (ctx.instr.op_count() > 1 && ctx.instr.op_kind(1) == iced_x86::OpKind::Register) {
    count = mm_read_source_or_zero(ctx, 1);
  } else if (ctx.instr.op_count() > 1 && ctx.instr.op_kind(1) != iced_x86::OpKind::Immediate8) {
    count = mm_read_source_or_zero(ctx, 1);
  } else {
    count = ctx.instr.immediate8();
  }
  const std::uint32_t masked = static_cast<std::uint32_t>(count & 0x3Fu);
  std::uint64_t out = lhs;
  for (std::size_t lane = 0; lane < 8; lane += sizeof(T)) {
    mm_lane_store(out, lane, fn(mm_lane_load<T>(lhs, lane), masked));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

#define KUBERA_MMX_SHIFT_IMM(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return mm_shift<type>(ctx, fn); }

KUBERA_MMX_SHIFT_IMM(PSLLW_MM_IMM8, std::uint16_t, mm_shift_left<std::uint16_t>)
KUBERA_MMX_SHIFT_IMM(PSLLD_MM_IMM8, std::uint32_t, mm_shift_left<std::uint32_t>)
KUBERA_MMX_SHIFT_IMM(PSLLQ_MM_IMM8, std::uint64_t, mm_shift_left<std::uint64_t>)
KUBERA_MMX_SHIFT_IMM(PSRLW_MM_IMM8, std::uint16_t, mm_shift_right_logical<std::uint16_t>)
KUBERA_MMX_SHIFT_IMM(PSRLD_MM_IMM8, std::uint32_t, mm_shift_right_logical<std::uint32_t>)
KUBERA_MMX_SHIFT_IMM(PSRLQ_MM_IMM8, std::uint64_t, mm_shift_right_logical<std::uint64_t>)
KUBERA_MMX_SHIFT_IMM(PSRAW_MM_IMM8, std::int16_t, mm_shift_right_arithmetic<std::int16_t>)
KUBERA_MMX_SHIFT_IMM(PSRAD_MM_IMM8, std::int32_t, mm_shift_right_arithmetic<std::int32_t>)

#define KUBERA_MMX_SHIFT_REG(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return mm_shift<type>(ctx, fn); }

KUBERA_MMX_SHIFT_REG(PSLLW_MM_MMM64, std::uint16_t, mm_shift_left<std::uint16_t>)
KUBERA_MMX_SHIFT_REG(PSLLD_MM_MMM64, std::uint32_t, mm_shift_left<std::uint32_t>)
KUBERA_MMX_SHIFT_REG(PSLLQ_MM_MMM64, std::uint64_t, mm_shift_left<std::uint64_t>)
KUBERA_MMX_SHIFT_REG(PSRLW_MM_MMM64, std::uint16_t, mm_shift_right_logical<std::uint16_t>)
KUBERA_MMX_SHIFT_REG(PSRLD_MM_MMM64, std::uint32_t, mm_shift_right_logical<std::uint32_t>)
KUBERA_MMX_SHIFT_REG(PSRLQ_MM_MMM64, std::uint64_t, mm_shift_right_logical<std::uint64_t>)
KUBERA_MMX_SHIFT_REG(PSRAW_MM_MMM64, std::int16_t, mm_shift_right_arithmetic<std::int16_t>)
KUBERA_MMX_SHIFT_REG(PSRAD_MM_MMM64, std::int32_t, mm_shift_right_arithmetic<std::int32_t>)

namespace {

template <typename Src>
inline std::uint64_t mm_pack(std::uint64_t dst, std::uint64_t src, Src (*saturate)(std::int64_t)) {
  constexpr std::size_t kSrcLane = sizeof(Src);
  constexpr std::size_t kCount = 8 / (kSrcLane * 2);
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    const std::size_t off = i * kSrcLane;
    Src v;
    std::memcpy(&v, reinterpret_cast<std::uint8_t*>(&dst) + off, kSrcLane);
    std::memcpy(reinterpret_cast<std::uint8_t*>(&out) + off, &v, kSrcLane);
    std::memcpy(&v, reinterpret_cast<std::uint8_t*>(&src) + off, kSrcLane);
    std::memcpy(reinterpret_cast<std::uint8_t*>(&out) + off + kSrcLane * kCount, &v, kSrcLane);
    (void)saturate;
  }
  return out;
}

template <typename Src, typename Dst>
inline Src mm_saturate_to(std::int64_t v) {
  constexpr std::int64_t lo = static_cast<std::int64_t>(std::numeric_limits<Dst>::min());
  constexpr std::int64_t hi = static_cast<std::int64_t>(std::numeric_limits<Dst>::max());
  return static_cast<Src>(v < lo ? lo : (v > hi ? hi : v));
}

}  // namespace

ExecutionResult handle_code_PACKSSWB_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = 0;
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::int8_t, std::int8_t>(v); };
  for (std::size_t i = 0; i < 4; ++i) {
    mm_lane_store(out, i, sat(mm_lane_load<std::int16_t>(dst, i * 2)));
    mm_lane_store(out, i + 4, sat(mm_lane_load<std::int16_t>(src, i * 2)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PACKSSDW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = 0;
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::int16_t, std::int16_t>(v); };
  for (std::size_t i = 0; i < 2; ++i) {
    mm_lane_store(out, i * 2, sat(mm_lane_load<std::int32_t>(dst, i * 4)));
    mm_lane_store(out, i * 2 + 4, sat(mm_lane_load<std::int32_t>(src, i * 4)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PACKUSWB_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = 0;
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::uint8_t, std::uint8_t>(v); };
  for (std::size_t i = 0; i < 4; ++i) {
    mm_lane_store(out, i, sat(mm_lane_load<std::uint16_t>(dst, i * 2)));
    mm_lane_store(out, i + 4, sat(mm_lane_load<std::uint16_t>(src, i * 2)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

namespace {

// PUNPCKL/H*: interleave the low (or high) elements of dst and src.
template <typename T>
ExecutionResult mm_unpack(ExecutionContext& ctx, bool high) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  constexpr std::size_t kCount = 8 / sizeof(T);
  constexpr std::size_t kHalf = kCount / 2;
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < kHalf; ++i) {
    const std::size_t idx = (high ? kHalf : 0) + i;
    mm_lane_store(out, (i * 2) * sizeof(T), mm_lane_load<T>(dst, idx * sizeof(T)));
    mm_lane_store(out, (i * 2 + 1) * sizeof(T), mm_lane_load<T>(src, idx * sizeof(T)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

ExecutionResult handle_code_PUNPCKLBW_MM_MMM32(ExecutionContext& ctx) { return mm_unpack<std::uint8_t>(ctx, false); }
ExecutionResult handle_code_PUNPCKLWD_MM_MMM32(ExecutionContext& ctx) { return mm_unpack<std::uint16_t>(ctx, false); }
ExecutionResult handle_code_PUNPCKLDQ_MM_MMM32(ExecutionContext& ctx) { return mm_unpack<std::uint32_t>(ctx, false); }
ExecutionResult handle_code_PUNPCKHBW_MM_MMM64(ExecutionContext& ctx) { return mm_unpack<std::uint8_t>(ctx, true); }
ExecutionResult handle_code_PUNPCKHWD_MM_MMM64(ExecutionContext& ctx) { return mm_unpack<std::uint16_t>(ctx, true); }
ExecutionResult handle_code_PUNPCKHDQ_MM_MMM64(ExecutionContext& ctx) { return mm_unpack<std::uint32_t>(ctx, true); }

ExecutionResult handle_code_PSHUFW_MM_MMM64_IMM8(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::Register) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const unsigned order = static_cast<unsigned>(ctx.instr.immediate8());
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const std::size_t sel = (order >> (i * 2)) & 3u;
    mm_lane_store(out, i * 2, mm_lane_load<std::uint16_t>(src, sel * 2));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PINSRW_MM_R32M16_IMM8(ExecutionContext& ctx) {
  bool ok = false;
  std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint16_t value = static_cast<std::uint16_t>(detail::read_operand(ctx, 1, 2, &ok));
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::size_t lane = static_cast<std::size_t>(ctx.instr.immediate8() & 3u);
  mm_lane_store(dst, lane * 2, value);
  return detail::write_operand(ctx, 0, dst, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PINSRW_MM_R64M16_IMM8(ExecutionContext& ctx) {
  return handle_code_PINSRW_MM_R32M16_IMM8(ctx);
}

ExecutionResult handle_code_PSHUFB_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  // SSSE3 MMX form: the shuffle control mask comes from the low 8 bytes of XMM0.
  const std::uint64_t control = static_cast<std::uint64_t>(ctx.state.vectors[0].value & mask(8));
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    const std::uint8_t sel = mm_lane_load<std::uint8_t>(control, i);
    const std::uint8_t value = (sel & 0x80u) != 0u ? std::uint8_t{0}
        : mm_lane_load<std::uint8_t>((i < 4) ? dst : src, sel & 7u);
    mm_lane_store(out, i, value);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

namespace {

template <typename T>
ExecutionResult mm_horizontal(ExecutionContext& ctx, bool add) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  constexpr std::size_t kCount = 8 / sizeof(T);
  std::uint64_t out = 0;
  if constexpr (sizeof(T) == 2) {
    auto op = [](std::int32_t a, std::int32_t b, bool is_add) { return is_add ? a + b : a - b; };
    for (std::size_t i = 0; i < kCount / 2; ++i) {
      const std::int32_t d0 = mm_lane_load<std::int16_t>(dst, i * 4);
      const std::int32_t d1 = mm_lane_load<std::int16_t>(dst, i * 4 + 2);
      mm_lane_store(out, i * 2, static_cast<std::int16_t>(op(d0, d1, add)));
    }
    for (std::size_t i = 0; i < kCount / 2; ++i) {
      const std::int32_t s0 = mm_lane_load<std::int16_t>(src, i * 4);
      const std::int32_t s1 = mm_lane_load<std::int16_t>(src, i * 4 + 2);
      mm_lane_store(out, (kCount / 2 + i) * 2, static_cast<std::int16_t>(op(s0, s1, add)));
    }
  } else {
    const std::int32_t d0 = mm_lane_load<std::int32_t>(dst, 0);
    const std::int32_t d1 = mm_lane_load<std::int32_t>(dst, 4);
    const std::int32_t s0 = mm_lane_load<std::int32_t>(src, 0);
    const std::int32_t s1 = mm_lane_load<std::int32_t>(src, 4);
    mm_lane_store(out, 0, add ? d0 + d1 : d0 - d1);
    mm_lane_store(out, 4, add ? s0 + s1 : s0 - s1);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

ExecutionResult handle_code_PHADDW_MM_MMM64(ExecutionContext& ctx) { return mm_horizontal<std::uint16_t>(ctx, true); }
ExecutionResult handle_code_PHADDD_MM_MMM64(ExecutionContext& ctx) { return mm_horizontal<std::uint32_t>(ctx, true); }
ExecutionResult handle_code_PHSUBW_MM_MMM64(ExecutionContext& ctx) { return mm_horizontal<std::uint16_t>(ctx, false); }
ExecutionResult handle_code_PHSUBD_MM_MMM64(ExecutionContext& ctx) { return mm_horizontal<std::uint32_t>(ctx, false); }

ExecutionResult handle_code_PHADDSW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::int16_t, std::int16_t>(v); };
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    mm_lane_store(out, i * 2, sat(static_cast<std::int64_t>(mm_lane_load<std::int16_t>(dst, i * 4)) + mm_lane_load<std::int16_t>(dst, i * 4 + 2)));
    mm_lane_store(out, i * 2 + 4, sat(static_cast<std::int64_t>(mm_lane_load<std::int16_t>(src, i * 4)) + mm_lane_load<std::int16_t>(src, i * 4 + 2)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PHSUBSW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::int16_t, std::int16_t>(v); };
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    mm_lane_store(out, i * 2, sat(static_cast<std::int64_t>(mm_lane_load<std::int16_t>(dst, i * 4)) - mm_lane_load<std::int16_t>(dst, i * 4 + 2)));
    mm_lane_store(out, i * 2 + 4, sat(static_cast<std::int64_t>(mm_lane_load<std::int16_t>(src, i * 4)) - mm_lane_load<std::int16_t>(src, i * 4 + 2)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PMADDUBSW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  auto sat = [](std::int64_t v) { return mm_saturate_to<std::int16_t, std::int16_t>(v); };
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const std::int32_t u = mm_lane_load<std::uint8_t>(dst, i * 2);
    const std::int32_t s = mm_lane_load<std::int8_t>(src, i * 2);
    const std::int32_t u2 = mm_lane_load<std::uint8_t>(dst, i * 2 + 1);
    const std::int32_t s2 = mm_lane_load<std::int8_t>(src, i * 2 + 1);
    mm_lane_store(out, i * 2, sat(static_cast<std::int64_t>(u * s) + (u2 * s2)));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_PMULHRSW_MM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const std::int32_t prod = mm_lane_load<std::int16_t>(dst, i * 2) * mm_lane_load<std::int16_t>(src, i * 2);
    mm_lane_store(out, i * 2, static_cast<std::int16_t>((prod + 0x4000) >> 15));
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

namespace {

template <typename T>
ExecutionResult mm_abs(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t out = dst;
  for (std::size_t lane = 0; lane < 8; lane += sizeof(T)) {
    const T v = mm_lane_load<T>(dst, lane);
    mm_lane_store(out, lane, v < 0 ? static_cast<T>(-static_cast<std::int64_t>(v)) : v);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

ExecutionResult handle_code_PABSB_MM_MMM64(ExecutionContext& ctx) { return mm_abs<std::int8_t>(ctx); }
ExecutionResult handle_code_PABSW_MM_MMM64(ExecutionContext& ctx) { return mm_abs<std::int16_t>(ctx); }
ExecutionResult handle_code_PABSD_MM_MMM64(ExecutionContext& ctx) { return mm_abs<std::int32_t>(ctx); }

namespace {

ExecutionResult mm_sign(ExecutionContext& ctx, std::size_t width) {
  bool ok = false;
  std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  for (std::size_t lane = 0; lane < 8; lane += width) {
    std::int64_t s = 0;
    if (width == 1) s = mm_lane_load<std::int8_t>(src, lane);
    else if (width == 2) s = mm_lane_load<std::int16_t>(src, lane);
    else s = mm_lane_load<std::int32_t>(src, lane);
    if (s < 0) {
      // negate each element (in place, wrapping)
      for (std::size_t b = 0; b < width; ++b) {
        const std::size_t byte = lane + b;
        const std::uint8_t byte_value = mm_lane_load<std::uint8_t>(dst, byte);
        mm_lane_store(dst, byte, static_cast<std::uint8_t>(~byte_value + 1u));
      }
    } else if (s == 0) {
      for (std::size_t b = 0; b < width; ++b) mm_lane_store(dst, lane + b, std::uint8_t{0});
    }
  }
  return detail::write_operand(ctx, 0, dst, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

ExecutionResult handle_code_PSIGNB_MM_MMM64(ExecutionContext& ctx) { return mm_sign(ctx, 1); }
ExecutionResult handle_code_PSIGNW_MM_MMM64(ExecutionContext& ctx) { return mm_sign(ctx, 2); }
ExecutionResult handle_code_PSIGND_MM_MMM64(ExecutionContext& ctx) { return mm_sign(ctx, 4); }

ExecutionResult handle_code_PALIGNR_MM_MMM64_IMM8(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t dst = detail::read_operand(ctx, 0, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const std::uint64_t src = mm_read_source(ctx, 1, ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  // Concatenate {src, dst} (src = high 8 bytes) and shift right by imm8 bytes.
  std::uint8_t joined[16];
  std::memcpy(joined, &dst, 8);
  std::memcpy(joined + 8, &src, 8);
  const unsigned shift = static_cast<unsigned>(ctx.instr.immediate8());
  std::uint64_t out = 0;
  if (shift < 8) {
    std::memcpy(&out, joined + shift, 8);
  } else if (shift < 16) {
    std::memcpy(&out, joined + shift, 16 - shift);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

// CVTPS2PI / CVTTPS2PI: convert the two low single-precision floats of the
// source to packed int32 in the MMX register.
namespace {

std::uint8_t mm_cw_to_rounding(std::uint16_t cw) {
  switch ((cw >> 10) & 0x3u) {
    case 0: return softfloat_round_near_even;
    case 1: return softfloat_round_min;
    case 2: return softfloat_round_max;
    default: return softfloat_round_minMag;
  }
}

ExecutionResult mm_cvtps2pi(ExecutionContext& ctx, bool truncate) {
  bool ok = false;
  std::uint8_t raw[8]{};
  if (ctx.instr.op_kind(1) == iced_x86::OpKind::Register) {
    const auto reg = ctx.instr.op_register(1);
    if (reg < iced_x86::Register::XMM0 || reg > iced_x86::Register::XMM15) {
      return detail::memory_fault(ctx, detail::memory_address(ctx));
    }
    const auto value = ctx.state.vectors[vector_index(reg)].value;
    for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((value >> (i * 8)) & 0xFFu);
  } else {
    const auto addr = detail::memory_address(ctx);
    if (!ctx.memory.read(addr, raw, 8)) return detail::memory_fault(ctx, addr);
  }
  const auto mode = truncate ? softfloat_round_minMag : mm_cw_to_rounding(ctx.state.get_x87_control_word());
  std::uint64_t out = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, raw + i * 4, 4);
    // SoftFloat's 8086-specialised f32_to_i32 returns the x86 integer
    // indefinite (0x80000000) for NaN/overflow, matching hardware.
    const std::int32_t converted = f32_to_i32(std::bit_cast<float32_t>(bits), mode, false);
    std::memcpy(reinterpret_cast<std::uint8_t*>(&out) + i * 4, &converted, 4);
  }
  return detail::write_operand(ctx, 0, out, 8) ? ExecutionResult{} : detail::memory_fault(ctx, detail::memory_address(ctx));
}

}  // namespace

ExecutionResult handle_code_CVTPS2PI_MM_XMMM64(ExecutionContext& ctx) { return mm_cvtps2pi(ctx, false); }
ExecutionResult handle_code_CVTTPS2PI_MM_XMMM64(ExecutionContext& ctx) { return mm_cvtps2pi(ctx, true); }

ExecutionResult handle_code_CVTPI2PS_XMM_MMM64(ExecutionContext& ctx) {
  bool ok = false;
  const std::uint64_t src = detail::read_operand(ctx, 1, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto dst_reg = ctx.instr.op_register(0);
  if (dst_reg < iced_x86::Register::XMM0 || dst_reg > iced_x86::Register::XMM15) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  big_uint out = ctx.state.vectors[vector_index(dst_reg)].value;
  for (std::size_t i = 0; i < 2; ++i) {
    const std::int32_t v = mm_lane_load<std::int32_t>(src, i * 4);
    const auto bits = i32_to_f32(v);
    out &= ~(big_uint(0xFFFFFFFFu) << (i * 32));
    out |= big_uint(bits.v) << (i * 32);
  }
  ctx.state.vectors[vector_index(dst_reg)].value = out;
  return {};
}

}  // namespace seven::handlers





