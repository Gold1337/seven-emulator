#include <gtest/gtest.h>

#include <cmath>

#include <iced_x86/code.hpp>
#include <iced_x86/instruction.hpp>
#include <iced_x86/register.hpp>

#include "kubera_test_support.hpp"

namespace {

using namespace kubera::test;

TEST(KuberaSimd, VpAndUsesExplicitSources) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with3(
      iced_x86::Code::VEX_Vpand_xmm_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      iced_x86::Register::XMM2).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vpand xmm0, xmm1, xmm2"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 0, 0xFFFF'FFFF'FFFF'FFFFull, 0xFFFF'FFFF'FFFF'FFFFull);
               set_xmm_u64(state, 1, 0xFF00'FF00'0F0F'0F0Full, 0xAAAA'5555'1234'5678ull);
               set_xmm_u64(state, 2, 0x0F0F'F0F0'FFFF'0000ull, 0xFFFF'0000'FFFF'0000ull);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x0F00'F000'0F0F'0000ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0xAAAA'0000'1234'0000ull);
             });
}

TEST(KuberaSimd, VaddssPreservesUpperLanesFromSrc1) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with3(
      iced_x86::Code::VEX_Vaddss_xmm_xmm_xmmm32,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      iced_x86::Register::XMM2).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vaddss xmm0, xmm1, xmm2"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state,
                             0,
                             std::bit_cast<std::uint32_t>(100.0f),
                             0xDEAD'BEEFu,
                             0xCAFEBABEu,
                             0x1357'2468u);
               set_xmm_u32x4(state,
                             1,
                             std::bit_cast<std::uint32_t>(1.5f),
                             0x1122'3344u,
                             0x5566'7788u,
                             0x99AA'BBCCu);
               set_xmm_u32x4(state,
                             2,
                             std::bit_cast<std::uint32_t>(2.25f),
                             0x0102'0304u,
                             0x0506'0708u,
                             0x090A'0B0Cu);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 3.75f);
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x1122'3344u);
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x5566'7788u);
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x99AA'BBCCu);
             });
}

TEST(KuberaSimd, SseCvtsi2ssConvertsAndPreservesUpperLanes) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvtsi2ss_xmm_rm64,
      iced_x86::Register::XMM0,
      iced_x86::Register::RAX).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvtsi2ss xmm0, rax"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 42;
               set_xmm_u32x4(state, 0, 0xDEAD'BEEFu, 0x1122'3344u, 0x5566'7788u, 0x99AA'BBCCu);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 42.0f);
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x1122'3344u);
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x5566'7788u);
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x99AA'BBCCu);
             });
}


TEST(KuberaSimd, Sse2Cvttsd2siTruncates) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttsd2si_r64_xmmm64,
      iced_x86::Register::RAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttsd2si rax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<double>(state, 1, 3.75);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 3u);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR64Truncates) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r64_xmmm32,
      iced_x86::Register::RAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si rax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<float>(state, 1, 3.75f);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 3u);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR32TruncatesTowardZero) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r32_xmmm32,
      iced_x86::Register::EAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si eax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<float>(state, 1, -3.75f);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0xFFFF'FFFDull);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR64FromMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r64_xmmm32,
      iced_x86::Register::RAX,
      iced_x86::MemoryOperand::with_base_displ_bcst_seg(iced_x86::Register::RCX, 0, false,
                                                        iced_x86::Register::None)).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si rax, dword ptr [rcx]"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory& memory) {
               state.gpr[1] = kBase + 0x800;
               const auto bits = std::bit_cast<std::uint32_t>(2.75f);
               ASSERT_TRUE(memory.write(kBase + 0x800, &bits, sizeof(bits)));
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 2u);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR64IndefiniteOnNaN) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r64_xmmm32,
      iced_x86::Register::RAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si rax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state, 1, 0x7FC0'0000u, 0, 0, 0);  // quiet NaN
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0x8000'0000'0000'0000ull);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR32IndefiniteOnOverflow) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r32_xmmm32,
      iced_x86::Register::EAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si eax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<float>(state, 1, 3.0e9f);  // fits float, exceeds int32
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0x8000'0000ull);
             });
}

TEST(KuberaSimd, Sse2Cvttss2siR64AcceptsInt32OverflowValues) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Cvttss2si_r64_xmmm32,
      iced_x86::Register::RAX,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttss2si rax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<float>(state, 1, 3.0e9f);  // exceeds int32, fits int64
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 3000000000ull);
             });
}

TEST(KuberaSimd, Sse2PextrwExtractsSelectedWord) {
  std::vector<std::uint8_t> bytes{0x66, 0x0F, 0xC5, 0xC1, 0x01};

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'FFFF'FFFFull;
               set_xmm_u32x4(state, 1, 0xABCD'5678u, 0, 0, 0);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0xABCDu);
             });
}


TEST(KuberaSimd, Sse3HaddpsProducesHorizontalSums) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Haddps_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "haddps xmm0, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state,
                             0,
                             std::bit_cast<std::uint32_t>(1.0f),
                             std::bit_cast<std::uint32_t>(2.0f),
                             std::bit_cast<std::uint32_t>(10.0f),
                             std::bit_cast<std::uint32_t>(20.0f));
               set_xmm_u32x4(state,
                             1,
                             std::bit_cast<std::uint32_t>(3.0f),
                             std::bit_cast<std::uint32_t>(4.0f),
                             std::bit_cast<std::uint32_t>(30.0f),
                             std::bit_cast<std::uint32_t>(40.0f));
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 3.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 1)), 30.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 2)), 7.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 3)), 70.0f);
             });
}

TEST(KuberaSimd, Sse42Crc32MatchesCastagnoliReference) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Crc32_r64_rm64,
      iced_x86::Register::RAX,
      iced_x86::Register::RBX).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "crc32 rax, rbx"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x1234'5678u;
               state.gpr[3] = 0x0123'4567'89AB'CDEFu;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               const auto expected = static_cast<std::uint64_t>(crc32c_update(0x1234'5678u, 0x0123'4567'89AB'CDEFu, 8));
               EXPECT_EQ(state.gpr[0], expected);
             });
}

namespace {

// XMM0 = 00 01 02 ... 0F; XMM1 = FIPS-197 AES-128 key bytes.
void setup_aes_state(seven::CpuState& state, seven::Memory&) {
  set_xmm_u64(state, 0, 0x0706'0504'0302'0100ull, 0x0F0E'0D0C'0B0A'0908ull);
  set_xmm_u64(state, 1, 0xA6D2'AE28'1615'7E2Bull, 0x3C4F'CF09'8815'F7ABull);
}

constexpr std::uint64_t kAesDstLow = 0x0706'0504'0302'0100ull;
constexpr std::uint64_t kAesDstHigh = 0x0F0E'0D0C'0B0A'0908ull;
constexpr std::uint64_t kAesKeyLow = 0xA6D2'AE28'1615'7E2Bull;
constexpr std::uint64_t kAesKeyHigh = 0x3C4F'CF09'8815'F7ABull;

}  // namespace

TEST(KuberaSimd, AesEncMatchesReferenceRound) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Aesenc_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aesenc xmm0, xmm1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: F7E1C30453491441606E532EE9482E1B
               EXPECT_EQ(xmm_u64(state, 0, 0), 0xF7E1'C304'5349'1441ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x606E'532E'E948'2E1Bull);
             });
}

TEST(KuberaSimd, AesEncLastMatchesReferenceRound) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Aesenclast_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aesenclast xmm0, xmm1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: 48157260DAAF79DD9B20624DF7B32017
               EXPECT_EQ(xmm_u64(state, 0, 0), 0xDD79'AFDA'6072'1548ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x1720'B3F7'4D62'209Bull);
             });
}

TEST(KuberaSimd, AesImcMatchesReference) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Aesimc_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aesimc xmm0, xmm1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: A2DC55D9073BC3A9DDC05FF7BA695C4E
               EXPECT_EQ(xmm_u64(state, 0, 0), 0xA2DC'55D9'073B'C3A9ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0xDDC0'5FF7'BA69'5C4Eull);
             });
}

TEST(KuberaSimd, AesKeyGenAssistMatchesReference) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with3(
      iced_x86::Code::Aeskeygenassist_xmm_xmmm128_imm8,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      static_cast<std::int32_t>(1)).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aeskeygenassist xmm0, xmm1, 1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: F35947F028AED2A66859C46309CF4F3C
               EXPECT_EQ(xmm_u64(state, 0, 0), 0xA6D2'AE28'F047'59F3ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x3C4F'CF09'63C4'5968ull);
             });
}

TEST(KuberaSimd, AesDecMatchesReferenceRound) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Aesdec_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aesdec xmm0, xmm1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: C8FBB7060A83DD386DD472CF3AE7B8BC
               EXPECT_EQ(xmm_u64(state, 0, 0), 0xC8FB'B706'0A83'DD38ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x6DD4'72CF'3AE7'B8BCull);
             });
}

TEST(KuberaSimd, AesDecLastMatchesReferenceRound) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with2(
      iced_x86::Code::Aesdeclast_xmm_xmmm128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "aesdeclast xmm0, xmm1"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: 3805A7182EB68D79E9EA8F88737FC114
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x3805'A718'2EB6'8D79ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0xE9EA'8F88'737F'C114ull);
             });
}

TEST(KuberaSimd, PclmulqdqMatchesReferenceProduct) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::Instruction::with3(
      iced_x86::Code::Pclmulqdq_xmm_xmmm128_imm8,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      static_cast<std::int32_t>(0x00)).value();
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "pclmulqdq xmm0, xmm1, 0"));

  run_single(bytes,
             setup_aes_state,
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               // Reference bytes: 002B28941244ED3B1A14243E9AE97303
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x3BED'4412'9428'2B00ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x0373'E99A'3E24'141Aull);
             });
}


}  // namespace
