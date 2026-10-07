#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include <iced_x86/code.hpp>
#include <iced_x86/instruction.hpp>

#include "seven/memory.hpp"
#include "seven/types.hpp"

namespace seven {

struct InstructionHookContext {
  CpuState& state;
  Memory& memory;
  const iced_x86::Instruction& instr;
  std::uint64_t next_rip;
};

enum class InstructionHookAction : std::uint8_t {
  continue_to_core,
  skip_core,
  stop,
};

struct InstructionHookResult {
  InstructionHookAction action = InstructionHookAction::continue_to_core;
  std::optional<ExecutionResult> stop_result;
};

struct StopHookEvent {
  CpuState& state;
  Memory& memory;
  ExecutionResult result;
  std::uint64_t fault_address = 0;
};

enum class FaultHookAction : std::uint8_t {
  stop,
  retry,
  restart_instruction,
};

struct FaultHookEvent {
  CpuState& state;
  Memory& memory;
  ExecutionResult result;
  std::uint64_t instruction_rip = 0;
  std::uint64_t fault_address = 0;
};

enum class TrapKind : std::uint8_t {
  syscall,
  cpuid,
  rdtsc,
  rdtscp,
  invalid_opcode,
  interrupt,
  rdrand,
  rdseed,
};

struct TrapHookContext {
  CpuState& state;
  Memory& memory;
  const iced_x86::Instruction& instr;
  std::uint64_t next_rip;
  TrapKind kind = TrapKind::invalid_opcode;
  // Vector of an internally-raised interrupt (e.g. the #DB single-step or
  // data-breakpoint fault, which has no INT instruction to decode). Only set
  // for TrapKind::interrupt hooks raised by the core itself; instruction-driven
  // interrupts leave it empty and hooks derive the vector from `instr`.
  std::optional<int> forced_interrupt;
};

enum class TrapHookAction : std::uint8_t {
  continue_to_core,
  handled,
  stop,
};

struct TrapHookResult {
  TrapHookAction action = TrapHookAction::continue_to_core;
  std::optional<ExecutionResult> stop_result;
};

using InstructionHook = std::function<InstructionHookResult(InstructionHookContext&)>;
using StopHook = std::function<void(const StopHookEvent&)>;
using FaultHook = std::function<FaultHookAction(const FaultHookEvent&)>;
using TrapHook = std::function<TrapHookResult(TrapHookContext&)>;

}  // namespace seven
