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

}  // namespace handlers
}  // namespace seven


