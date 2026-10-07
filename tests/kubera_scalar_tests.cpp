#include <gtest/gtest.h>

#include "kubera_test_support.hpp"

namespace {

using namespace kubera::test;

TEST(KuberaScalar, ImulAndMulFlagSemantics) {
  run_single(seven::parse_hex_bytes("6B C1 05"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'1234'5678ull;
               state.gpr[1] = 7;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 35u);
               EXPECT_EQ(state.rip, kBase + 3);
             });

  run_single(seven::parse_hex_bytes("F6 E3"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x81;
               state.gpr[3] = 0x02;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0102u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagOF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagAF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagZF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagSF, 0u);
               EXPECT_EQ((state.rflags & seven::kFlagPF) != 0u, seven::even_parity(0x02));
             });
}

TEST(KuberaScalar, DivIdivAndDivideErrorSemantics) {
  run_single(seven::parse_hex_bytes("F6 F3"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x0014;
               state.gpr[3] = 0x03;
               state.rflags = 0x202 | seven::kFlagCF | seven::kFlagPF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0206u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagPF, 0u);
             });

  run_single(seven::parse_hex_bytes("F6 FB"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFA;
               state.gpr[3] = 0xFD;
               state.rflags = 0x202 | seven::kFlagOF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0002u);
               EXPECT_NE(state.rflags & seven::kFlagOF, 0u);
             });

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  const auto bytes = seven::parse_hex_bytes("F6 F3");
  write_bytes(memory, kBase, bytes);
  state.gpr[0] = 0x0014;
  state.gpr[3] = 0x00;
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::divide_error);
}

TEST(KuberaScalar, IncAndShiftEdgeCases) {
  run_single(seven::parse_hex_bytes("FF C0"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'FFFF'FFFFull;
               state.rflags = 0x202 | seven::kFlagCF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
             });

  run_single(seven::parse_hex_bytes("C0 E0 08"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x81;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFu, 0u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
             });
}

TEST(KuberaScalar, WritablePagesAreReadable) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = 0x4000;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
  const auto code = seven::parse_hex_bytes("8B 03");
  const std::uint32_t value = 0x12345678u;
  write_bytes(memory, kBase, code);
  EXPECT_TRUE(memory.write_unchecked(0x4000, value));
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(static_cast<std::uint32_t>(state.gpr[0]), value);
}


TEST(KuberaScalar, RtlInterlockedFlushSListSequenceMatchesExpected) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kBase + 0x800 - 8;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, seven::kMemoryPermissionAll);
  const auto code = seven::parse_hex_bytes("0F 0D 09 53 4C 8B D1 48 8B 01 48 8B 51 08 33 C9 48 8B D8 66 33 DB F0 49 0F C7 0A 75 F1 48 8B C2 24 F0 5B C3");
  write_bytes(memory, kBase, code);
  const std::uint64_t return_target = kBase + static_cast<std::uint64_t>(code.size());
  EXPECT_TRUE(memory.write_unchecked(state.gpr[4], return_target));
  state.gpr[0] = 0x1111222233334444ull;
  state.gpr[1] = 0x4000ull;
  state.gpr[2] = 0xAAAABBBBCCCCDDD5ull;
  state.gpr[3] = 0xDEADBEEFCAFEBABEu;
  const std::uint64_t old_low = 0x1234567890ABCDEFull;
  const std::uint64_t old_high = 0xAAAABBBBCCCCDDD5ull;
  EXPECT_TRUE(memory.write_unchecked(0x4000, old_low));
  EXPECT_TRUE(memory.write_unchecked(0x4008, old_high));
  bool reached_return = false;
  for (int i = 0; i < 16; ++i) {
    const auto result = executor.step(state, memory);
    ASSERT_EQ(result.reason, seven::StopReason::none);
    if (state.rip == return_target) {
      reached_return = true;
      break;
    }
  }
  ASSERT_TRUE(reached_return);
  const std::uint64_t expected_low = old_low & ~0xFFFFull;
  std::uint64_t actual_low = 0;
  std::uint64_t actual_high = 0;
  EXPECT_TRUE(memory.read_unchecked(0x4000, actual_low));
  EXPECT_TRUE(memory.read_unchecked(0x4008, actual_high));
  EXPECT_EQ(actual_low, expected_low);
  EXPECT_EQ(actual_high, 0u);
  EXPECT_EQ(state.gpr[0], old_high & ~0xFull);
  EXPECT_EQ(state.gpr[3], 0xDEADBEEFCAFEBABEu);
}


TEST(KuberaScalar, RtlInterlockedPushListSListSequenceMatchesExpected) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kBase + 0x800 - 8;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, seven::kMemoryPermissionAll);
  memory.map(0x5000, 0x1000, seven::kMemoryPermissionAll);
  const auto code = seven::parse_hex_bytes("0F 0D 09 53 48 8B 01 4C 8B D9 48 8B CA 49 8B 53 08 4C 8B D2 41 80 E2 F0 4D 89 10 48 8D 98 00 00 02 00 66 42 8D 1C 08 F0 49 0F C7 0B 75 E3 49 8B C2 5B C3");
  write_bytes(memory, kBase, code);
  const std::uint64_t return_target = kBase + static_cast<std::uint64_t>(code.size());
  EXPECT_TRUE(memory.write_unchecked(state.gpr[4], return_target));
  const std::uint64_t old_low = 0x0000000000010002ull;
  const std::uint64_t old_head = 0x1234567890ABCDEFull;
  const std::uint64_t new_head = 0x5000ull;
  const std::uint64_t out_ptr = 0x5010ull;
  EXPECT_TRUE(memory.write_unchecked(0x4000, old_low));
  EXPECT_TRUE(memory.write_unchecked(0x4008, old_head));
  state.gpr[1] = 0x4000ull;
  state.gpr[2] = new_head;
  state.gpr[8] = out_ptr;
  state.gpr[9] = 3;
  bool reached_return = false;
  for (int i = 0; i < 16; ++i) {
    const auto result = executor.step(state, memory);
    ASSERT_EQ(result.reason, seven::StopReason::none);
    if (state.rip == return_target) {
      reached_return = true;
      break;
    }
  }
  ASSERT_TRUE(reached_return);
  std::uint64_t actual_low = 0;
  std::uint64_t actual_head = 0;
  std::uint64_t old_head_out = 0;
  EXPECT_TRUE(memory.read_unchecked(0x4000, actual_low));
  EXPECT_TRUE(memory.read_unchecked(0x4008, actual_head));
  EXPECT_TRUE(memory.read_unchecked(out_ptr, old_head_out));
  EXPECT_EQ(actual_head, new_head);
  EXPECT_EQ(actual_low, ((old_low + 0x20000ull) & ~0xFFFFull) | ((old_low + 3) & 0xFFFFull));
  EXPECT_EQ(old_head_out, old_head & ~0xFull);
  EXPECT_EQ(state.gpr[0], old_head & ~0xFull);
}


TEST(KuberaScalar, OneOperandImul64HandlesNegativeHighHalf) {
  run_single(seven::parse_hex_bytes("48 F7 E9"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = static_cast<std::uint64_t>(-5ll);
               state.gpr[1] = 7;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(static_cast<std::int64_t>(state.gpr[0]), -35ll);
               EXPECT_EQ(static_cast<std::int64_t>(state.gpr[2]), -1ll);
             });
}


TEST(KuberaScalar, JneRel32BranchesWhenZfClear) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 85 04 00 00 00 90 90 90 90"));
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 10);
}


TEST(KuberaScalar, PrefetchDoesNotFaultUnmappedMemory) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[0] = 0xDEAD'BEEF0000ull;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 18 08"));
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 3);
}


TEST(KuberaScalar, FxrstorReadsWritablePages) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[0] = 0x4000;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F AE 08"));
  std::array<std::byte, 512> fx{};
  EXPECT_TRUE(memory.write_unchecked(0x4000, fx.data(), fx.size()));
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
}


TEST(KuberaScalar, RdsspReportsNoShadowStack) {
  run_single(seven::parse_hex_bytes("F3 48 0F 1E CA"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[2] = 0xFFFF'FFFF'FFFF'FFFFull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[2], 0u);
             });
}

TEST(KuberaScalar, RdrandR32SetsFlagsAndZeroExtends) {
  run_single(seven::parse_hex_bytes("0F C7 F0"),  // rdrand eax
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xDEAD'BEEF'1111'1111ull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagOF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagSF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagZF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagAF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagPF, 0u);
               EXPECT_EQ(state.gpr[0] >> 32, 0u);
             });
}

TEST(KuberaScalar, RdrandR16PreservesUpperBits) {
  run_single(seven::parse_hex_bytes("66 0F C7 F0"),  // rdrand ax
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x1111'2222'3333'4444ull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_EQ(state.gpr[0] >> 16, 0x1111'2222'3333ull);
             });
}

TEST(KuberaScalar, RdrandR64SetsFlags) {
  run_single(seven::parse_hex_bytes("48 0F C7 F0"),  // rdrand rax
             [](seven::CpuState& state, seven::Memory&) {
               state.rflags = seven::kFlagZF | seven::kFlagSF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagZF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagSF, 0u);
             });
}

TEST(KuberaScalar, RdseedR32SetsFlagsAndZeroExtends) {
  run_single(seven::parse_hex_bytes("0F C7 F8"),  // rdseed eax
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xDEAD'BEEF'1111'1111ull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagOF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagZF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagSF, 0u);
               EXPECT_EQ(state.gpr[0] >> 32, 0u);
             });
}

TEST(KuberaScalar, RdrandTrapHookOverridesValue) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 0F C7 F0"));  // rdrand rax
  const auto hook_id = executor.add_trap_hook(seven::TrapKind::rdrand, [](seven::TrapHookContext& ctx) {
    ctx.state.gpr[0] = 0x1234'5678'9ABC'DEF0ull;
    seven::TrapHookResult result{};
    result.action = seven::TrapHookAction::handled;
    return result;
  });
  ASSERT_NE(hook_id, 0u);
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 0x1234'5678'9ABC'DEF0ull);
  EXPECT_EQ(state.rip, kBase + 4);
}

TEST(KuberaScalar, Ud2RaisesInvalidOpcodeFaultWithoutHook) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 0B"));  // ud2

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::invalid_opcode);
  EXPECT_EQ(state.rip, kBase);  // faulting instruction does not retire
}

TEST(KuberaScalar, Ud2RaisesInvalidOpcodeTrapHook) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 0B"));  // ud2

  const auto hook_id = executor.add_trap_hook(seven::TrapKind::invalid_opcode, [](seven::TrapHookContext& ctx) {
    ctx.state.gpr[0] = 0xC000'001Dull;  // what a host would report as the exception code
    seven::TrapHookResult result{};
    result.action = seven::TrapHookAction::handled;
    return result;
  });
  ASSERT_NE(hook_id, 0u);

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 0xC000'001Dull);
  EXPECT_EQ(state.rip, kBase + 2);  // handled trap advances past the ud2
}

TEST(KuberaScalar, RdseedTrapHookOverridesValue) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 0F C7 F8"));  // rdseed rax
  const auto hook_id = executor.add_trap_hook(seven::TrapKind::rdseed, [](seven::TrapHookContext& ctx) {
    ctx.state.gpr[0] = 0x0FED'CBA9'8765'4321ull;
    seven::TrapHookResult result{};
    result.action = seven::TrapHookAction::handled;
    return result;
  });
  ASSERT_NE(hook_id, 0u);
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 0x0FED'CBA9'8765'4321ull);
  EXPECT_EQ(state.rip, kBase + 4);
}

TEST(KuberaScalar, IretqRestoresRipCsFlagsRspAndSs) {
  // IRETQ pops five qword slots: RIP, CS, RFLAGS, RSP, SS. ntdll's user-mode
  // exception restore (RtlRestoreContext) ends in iretq and relies on the
  // RSP/SS fields being honoured.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CF"));  // iretq

  // Build the iretq frame at 0x4000.
  const std::uint64_t new_rsp = 0x4000 + 0x100;
  const std::array<std::uint64_t, 5> frame{
      0x1234'5678'9ABC'DEF0ull,  // RIP
      0x33,                      // CS (qword slot, low 16 used)
      0x246,                     // RFLAGS
      new_rsp,                   // RSP
      0x2B};                     // SS (qword slot, low 16 used)
  ASSERT_TRUE(memory.write(0x4000, frame.data(), frame.size() * 8));
  state.gpr[4] = 0x4000;

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, 0x1234'5678'9ABC'DEF0ull);
  EXPECT_EQ(state.sreg[1], 0x33u);             // CS
  EXPECT_EQ(state.sreg[2], 0x2Bu);             // SS
  EXPECT_EQ(state.rflags & 0xFFFFFFFFull, 0x246u);
  EXPECT_EQ(state.gpr[4], new_rsp);            // RSP restored from the frame, not old rsp + slots
}

TEST(KuberaScalar, IretqIgnoresUpperBitsOfCsAndSsSlots) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CF"));  // iretq

  // ntdll's restore stub writes only the low word into the CS/SS slots; the
  // upper bytes are stale stack data and must be ignored on load.
  const std::array<std::uint64_t, 5> frame{
      kBase + 0x20,                    // RIP
      0xAAAA'AAAA'AAAA'0033ull,        // CS slot with garbage upper bits
      0x202,                           // RFLAGS
      0x4000 + 0x80,                   // RSP
      0xBBBB'BEBB'BBBB'B2B2ull};       // SS slot with garbage upper bits
  ASSERT_TRUE(memory.write(0x4000, frame.data(), frame.size() * 8));
  state.gpr[4] = 0x4000;

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 0x20);
  EXPECT_EQ(state.sreg[1], 0x33u);
  EXPECT_EQ(state.sreg[2], 0xB2B2u & 0xFFFFu);
  EXPECT_EQ(state.gpr[4], 0x4000 + 0x80);
}

}  // namespace
