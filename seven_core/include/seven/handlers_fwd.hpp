#pragma once

#include "seven/executor.hpp"

namespace seven {
namespace handlers {

#define KUBERA_CODE(code, name) ExecutionResult handle_code_##name(ExecutionContext& ctx);
#include "seven/handled_codes.def"
#undef KUBERA_CODE

// RDRAND/RDSEED defaults used by the core when no TrapKind::rdrand/rdseed
// trap hook handles the instruction: writes a random value of `width` bytes
// to `dest` (host RDRAND/RDSEED instruction on x86 hosts, software RNG
// elsewhere) and sets EFLAGS per the SDM (CF = value returned, OF/SF/ZF/AF/PF
// cleared).
void emulate_rdrand_defaults(CpuState& state, iced_x86::Register dest, std::size_t width);
void emulate_rdseed_defaults(CpuState& state, iced_x86::Register dest, std::size_t width);

// Deterministic CPUID profile (see the definition for the rationale and the
// masked-off bits). This is BOTH the core's built-in CPUID behaviour (used
// when no TrapKind::cpuid hook handles the instruction) and the defaults a
// frontend applies after its own CPUID hook ran without skipping the
// instruction, so every entry point presents the same feature set.
void emulate_cpuid_defaults(CpuState& state);

}  // namespace handlers
}  // namespace seven


