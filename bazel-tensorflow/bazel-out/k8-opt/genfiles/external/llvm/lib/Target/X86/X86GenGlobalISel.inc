/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Global Instruction Selector for the X86 target                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_GLOBALISEL_PREDICATE_BITSET
const unsigned MAX_SUBTARGET_PREDICATES = 114;
using PredicateBitset = llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;
#endif // ifdef GET_GLOBALISEL_PREDICATE_BITSET

#ifdef GET_GLOBALISEL_TEMPORARIES_DECL
  mutable MatcherState State;
  typedef ComplexRendererFns(X86InstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
  typedef void(X86InstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr&) const;
  const ISelInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ISelInfo;
  static X86InstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
  static X86InstructionSelector::CustomRendererFn CustomRenderers[];
  bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
  bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
  bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
  const int64_t *getMatchTable() const override;
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

#ifdef GET_GLOBALISEL_TEMPORARIES_INIT
, State(0),
ISelInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT

#ifdef GET_GLOBALISEL_IMPL
// Bits for subtarget features that participate in instruction matching.
enum SubtargetFeatureBits : uint8_t {
  Feature_TruePredicateBit = 48,
  Feature_HasCMovBit = 21,
  Feature_NoCMovBit = 100,
  Feature_HasMMXBit = 81,
  Feature_Has3DNowBit = 83,
  Feature_HasSSE1Bit = 35,
  Feature_UseSSE1Bit = 41,
  Feature_HasSSE2Bit = 36,
  Feature_UseSSE2Bit = 42,
  Feature_HasSSE3Bit = 27,
  Feature_UseSSE3Bit = 51,
  Feature_HasSSSE3Bit = 82,
  Feature_UseSSSE3Bit = 52,
  Feature_NoSSE41Bit = 103,
  Feature_UseSSE41Bit = 49,
  Feature_HasSSE42Bit = 56,
  Feature_UseSSE42Bit = 55,
  Feature_HasSSE4ABit = 65,
  Feature_NoAVXBit = 61,
  Feature_HasAVXBit = 43,
  Feature_HasAVX2Bit = 39,
  Feature_HasAVX1OnlyBit = 37,
  Feature_HasAVX512Bit = 69,
  Feature_UseAVXBit = 40,
  Feature_UseAVX2Bit = 67,
  Feature_NoAVX512Bit = 32,
  Feature_HasCDIBit = 73,
  Feature_HasVPOPCNTDQBit = 77,
  Feature_HasERIBit = 76,
  Feature_HasDQIBit = 71,
  Feature_NoDQIBit = 53,
  Feature_HasBWIBit = 72,
  Feature_NoBWIBit = 50,
  Feature_HasVLXBit = 70,
  Feature_NoVLXBit = 31,
  Feature_NoVLX_Or_NoBWIBit = 47,
  Feature_NoVLX_Or_NoDQIBit = 105,
  Feature_HasVNNIBit = 79,
  Feature_HasBITALGBit = 80,
  Feature_HasPOPCNTBit = 54,
  Feature_HasAESBit = 58,
  Feature_HasVAESBit = 60,
  Feature_NoVLX_Or_NoVAESBit = 59,
  Feature_HasFXSRBit = 28,
  Feature_HasXSAVEBit = 89,
  Feature_HasXSAVEOPTBit = 90,
  Feature_HasXSAVECBit = 91,
  Feature_HasXSAVESBit = 92,
  Feature_HasPCLMULBit = 62,
  Feature_NoVLX_Or_NoVPCLMULQDQBit = 63,
  Feature_HasVPCLMULQDQBit = 64,
  Feature_HasGFNIBit = 68,
  Feature_HasFMABit = 29,
  Feature_HasFMA4Bit = 33,
  Feature_NoFMA4Bit = 30,
  Feature_HasXOPBit = 34,
  Feature_HasTBMBit = 9,
  Feature_NoTBMBit = 112,
  Feature_HasLWPBit = 10,
  Feature_HasMOVBEBit = 3,
  Feature_HasRDRANDBit = 4,
  Feature_HasF16CBit = 66,
  Feature_HasFSGSBaseBit = 93,
  Feature_HasLZCNTBit = 6,
  Feature_HasBMIBit = 7,
  Feature_HasBMI2Bit = 8,
  Feature_NoBMI2Bit = 113,
  Feature_HasVBMIBit = 74,
  Feature_HasVBMI2Bit = 78,
  Feature_HasIFMABit = 75,
  Feature_HasRTMBit = 87,
  Feature_HasADXBit = 20,
  Feature_HasSHABit = 57,
  Feature_HasRDSEEDBit = 5,
  Feature_HasSSEPrefetchBit = 45,
  Feature_NoSSEPrefetchBit = 84,
  Feature_HasPrefetchWBit = 85,
  Feature_HasPREFETCHWT1Bit = 86,
  Feature_HasLAHFSAHFBit = 2,
  Feature_HasMWAITXBit = 11,
  Feature_HasCLZEROBit = 15,
  Feature_HasCLDEMOTEBit = 18,
  Feature_HasMOVDIRIBit = 13,
  Feature_HasMOVDIR64BBit = 14,
  Feature_HasPTWRITEBit = 96,
  Feature_FPStackf32Bit = 25,
  Feature_FPStackf64Bit = 26,
  Feature_HasCLFLUSHOPTBit = 16,
  Feature_HasCLWBBit = 17,
  Feature_HasWBNOINVDBit = 88,
  Feature_HasRDPIDBit = 95,
  Feature_HasWAITPKGBit = 12,
  Feature_HasINVPCIDBit = 94,
  Feature_HasCmpxchg16bBit = 101,
  Feature_Not64BitModeBit = 0,
  Feature_In64BitModeBit = 1,
  Feature_IsLP64Bit = 98,
  Feature_NotLP64Bit = 97,
  Feature_NotWin64WithoutFPBit = 99,
  Feature_IsPS4Bit = 107,
  Feature_NotPS4Bit = 106,
  Feature_KernelCodeBit = 108,
  Feature_NearDataBit = 110,
  Feature_IsNotPICBit = 109,
  Feature_OptForSizeBit = 44,
  Feature_OptForMinSizeBit = 38,
  Feature_OptForSpeedBit = 104,
  Feature_UseIncDecBit = 19,
  Feature_CallImmAddrBit = 111,
  Feature_FavorMemIndirectCallBit = 22,
  Feature_HasFastSHLDRotateBit = 102,
  Feature_HasMFenceBit = 46,
  Feature_UseRetpolineBit = 24,
  Feature_NotUseRetpolineBit = 23,
};

PredicateBitset X86InstructionSelector::
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const {
  PredicateBitset Features;
  if (true)
    Features[Feature_TruePredicateBit] = 1;
  if (Subtarget->hasCMov())
    Features[Feature_HasCMovBit] = 1;
  if (!Subtarget->hasCMov())
    Features[Feature_NoCMovBit] = 1;
  if (Subtarget->hasMMX())
    Features[Feature_HasMMXBit] = 1;
  if (Subtarget->has3DNow())
    Features[Feature_Has3DNowBit] = 1;
  if (Subtarget->hasSSE1())
    Features[Feature_HasSSE1Bit] = 1;
  if (Subtarget->hasSSE1() && !Subtarget->hasAVX())
    Features[Feature_UseSSE1Bit] = 1;
  if (Subtarget->hasSSE2())
    Features[Feature_HasSSE2Bit] = 1;
  if (Subtarget->hasSSE2() && !Subtarget->hasAVX())
    Features[Feature_UseSSE2Bit] = 1;
  if (Subtarget->hasSSE3())
    Features[Feature_HasSSE3Bit] = 1;
  if (Subtarget->hasSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSE3Bit] = 1;
  if (Subtarget->hasSSSE3())
    Features[Feature_HasSSSE3Bit] = 1;
  if (Subtarget->hasSSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSSE3Bit] = 1;
  if (!Subtarget->hasSSE41())
    Features[Feature_NoSSE41Bit] = 1;
  if (Subtarget->hasSSE41() && !Subtarget->hasAVX())
    Features[Feature_UseSSE41Bit] = 1;
  if (Subtarget->hasSSE42())
    Features[Feature_HasSSE42Bit] = 1;
  if (Subtarget->hasSSE42() && !Subtarget->hasAVX())
    Features[Feature_UseSSE42Bit] = 1;
  if (Subtarget->hasSSE4A())
    Features[Feature_HasSSE4ABit] = 1;
  if (!Subtarget->hasAVX())
    Features[Feature_NoAVXBit] = 1;
  if (Subtarget->hasAVX())
    Features[Feature_HasAVXBit] = 1;
  if (Subtarget->hasAVX2())
    Features[Feature_HasAVX2Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX2())
    Features[Feature_HasAVX1OnlyBit] = 1;
  if (Subtarget->hasAVX512())
    Features[Feature_HasAVX512Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX512())
    Features[Feature_UseAVXBit] = 1;
  if (Subtarget->hasAVX2() && !Subtarget->hasAVX512())
    Features[Feature_UseAVX2Bit] = 1;
  if (!Subtarget->hasAVX512())
    Features[Feature_NoAVX512Bit] = 1;
  if (Subtarget->hasCDI())
    Features[Feature_HasCDIBit] = 1;
  if (Subtarget->hasVPOPCNTDQ())
    Features[Feature_HasVPOPCNTDQBit] = 1;
  if (Subtarget->hasERI())
    Features[Feature_HasERIBit] = 1;
  if (Subtarget->hasDQI())
    Features[Feature_HasDQIBit] = 1;
  if (!Subtarget->hasDQI())
    Features[Feature_NoDQIBit] = 1;
  if (Subtarget->hasBWI())
    Features[Feature_HasBWIBit] = 1;
  if (!Subtarget->hasBWI())
    Features[Feature_NoBWIBit] = 1;
  if (Subtarget->hasVLX())
    Features[Feature_HasVLXBit] = 1;
  if (!Subtarget->hasVLX())
    Features[Feature_NoVLXBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasBWI())
    Features[Feature_NoVLX_Or_NoBWIBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasDQI())
    Features[Feature_NoVLX_Or_NoDQIBit] = 1;
  if (Subtarget->hasVNNI())
    Features[Feature_HasVNNIBit] = 1;
  if (Subtarget->hasBITALG())
    Features[Feature_HasBITALGBit] = 1;
  if (Subtarget->hasPOPCNT())
    Features[Feature_HasPOPCNTBit] = 1;
  if (Subtarget->hasAES())
    Features[Feature_HasAESBit] = 1;
  if (Subtarget->hasVAES())
    Features[Feature_HasVAESBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVAES())
    Features[Feature_NoVLX_Or_NoVAESBit] = 1;
  if (Subtarget->hasFXSR())
    Features[Feature_HasFXSRBit] = 1;
  if (Subtarget->hasXSAVE())
    Features[Feature_HasXSAVEBit] = 1;
  if (Subtarget->hasXSAVEOPT())
    Features[Feature_HasXSAVEOPTBit] = 1;
  if (Subtarget->hasXSAVEC())
    Features[Feature_HasXSAVECBit] = 1;
  if (Subtarget->hasXSAVES())
    Features[Feature_HasXSAVESBit] = 1;
  if (Subtarget->hasPCLMUL())
    Features[Feature_HasPCLMULBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVPCLMULQDQ())
    Features[Feature_NoVLX_Or_NoVPCLMULQDQBit] = 1;
  if (Subtarget->hasVPCLMULQDQ())
    Features[Feature_HasVPCLMULQDQBit] = 1;
  if (Subtarget->hasGFNI())
    Features[Feature_HasGFNIBit] = 1;
  if (Subtarget->hasFMA())
    Features[Feature_HasFMABit] = 1;
  if (Subtarget->hasFMA4())
    Features[Feature_HasFMA4Bit] = 1;
  if (!Subtarget->hasFMA4())
    Features[Feature_NoFMA4Bit] = 1;
  if (Subtarget->hasXOP())
    Features[Feature_HasXOPBit] = 1;
  if (Subtarget->hasTBM())
    Features[Feature_HasTBMBit] = 1;
  if (!Subtarget->hasTBM())
    Features[Feature_NoTBMBit] = 1;
  if (Subtarget->hasLWP())
    Features[Feature_HasLWPBit] = 1;
  if (Subtarget->hasMOVBE())
    Features[Feature_HasMOVBEBit] = 1;
  if (Subtarget->hasRDRAND())
    Features[Feature_HasRDRANDBit] = 1;
  if (Subtarget->hasF16C())
    Features[Feature_HasF16CBit] = 1;
  if (Subtarget->hasFSGSBase())
    Features[Feature_HasFSGSBaseBit] = 1;
  if (Subtarget->hasLZCNT())
    Features[Feature_HasLZCNTBit] = 1;
  if (Subtarget->hasBMI())
    Features[Feature_HasBMIBit] = 1;
  if (Subtarget->hasBMI2())
    Features[Feature_HasBMI2Bit] = 1;
  if (!Subtarget->hasBMI2())
    Features[Feature_NoBMI2Bit] = 1;
  if (Subtarget->hasVBMI())
    Features[Feature_HasVBMIBit] = 1;
  if (Subtarget->hasVBMI2())
    Features[Feature_HasVBMI2Bit] = 1;
  if (Subtarget->hasIFMA())
    Features[Feature_HasIFMABit] = 1;
  if (Subtarget->hasRTM())
    Features[Feature_HasRTMBit] = 1;
  if (Subtarget->hasADX())
    Features[Feature_HasADXBit] = 1;
  if (Subtarget->hasSHA())
    Features[Feature_HasSHABit] = 1;
  if (Subtarget->hasRDSEED())
    Features[Feature_HasRDSEEDBit] = 1;
  if (Subtarget->hasSSEPrefetch())
    Features[Feature_HasSSEPrefetchBit] = 1;
  if (!Subtarget->hasSSEPrefetch())
    Features[Feature_NoSSEPrefetchBit] = 1;
  if (Subtarget->hasPRFCHW())
    Features[Feature_HasPrefetchWBit] = 1;
  if (Subtarget->hasPREFETCHWT1())
    Features[Feature_HasPREFETCHWT1Bit] = 1;
  if (Subtarget->hasLAHFSAHF())
    Features[Feature_HasLAHFSAHFBit] = 1;
  if (Subtarget->hasMWAITX())
    Features[Feature_HasMWAITXBit] = 1;
  if (Subtarget->hasCLZERO())
    Features[Feature_HasCLZEROBit] = 1;
  if (Subtarget->hasCLDEMOTE())
    Features[Feature_HasCLDEMOTEBit] = 1;
  if (Subtarget->hasMOVDIRI())
    Features[Feature_HasMOVDIRIBit] = 1;
  if (Subtarget->hasMOVDIR64B())
    Features[Feature_HasMOVDIR64BBit] = 1;
  if (Subtarget->hasPTWRITE())
    Features[Feature_HasPTWRITEBit] = 1;
  if (!Subtarget->hasSSE1())
    Features[Feature_FPStackf32Bit] = 1;
  if (!Subtarget->hasSSE2())
    Features[Feature_FPStackf64Bit] = 1;
  if (Subtarget->hasCLFLUSHOPT())
    Features[Feature_HasCLFLUSHOPTBit] = 1;
  if (Subtarget->hasCLWB())
    Features[Feature_HasCLWBBit] = 1;
  if (Subtarget->hasWBNOINVD())
    Features[Feature_HasWBNOINVDBit] = 1;
  if (Subtarget->hasRDPID())
    Features[Feature_HasRDPIDBit] = 1;
  if (Subtarget->hasWAITPKG())
    Features[Feature_HasWAITPKGBit] = 1;
  if (Subtarget->hasINVPCID())
    Features[Feature_HasINVPCIDBit] = 1;
  if (Subtarget->hasCmpxchg16b())
    Features[Feature_HasCmpxchg16bBit] = 1;
  if (!Subtarget->is64Bit())
    Features[Feature_Not64BitModeBit] = 1;
  if (Subtarget->is64Bit())
    Features[Feature_In64BitModeBit] = 1;
  if (Subtarget->isTarget64BitLP64())
    Features[Feature_IsLP64Bit] = 1;
  if (!Subtarget->isTarget64BitLP64())
    Features[Feature_NotLP64Bit] = 1;
  if (Subtarget->isTargetPS4())
    Features[Feature_IsPS4Bit] = 1;
  if (!Subtarget->isTargetPS4())
    Features[Feature_NotPS4Bit] = 1;
  if (TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_KernelCodeBit] = 1;
  if (TM.getCodeModel() == CodeModel::Small ||TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_NearDataBit] = 1;
  if (!TM.isPositionIndependent())
    Features[Feature_IsNotPICBit] = 1;
  if (Subtarget->isLegalToCallImmediateAddr())
    Features[Feature_CallImmAddrBit] = 1;
  if (!Subtarget->slowTwoMemOps())
    Features[Feature_FavorMemIndirectCallBit] = 1;
  if (Subtarget->hasFastSHLDRotate())
    Features[Feature_HasFastSHLDRotateBit] = 1;
  if (Subtarget->hasMFence())
    Features[Feature_HasMFenceBit] = 1;
  if (Subtarget->useRetpoline())
    Features[Feature_UseRetpolineBit] = 1;
  if (!Subtarget->useRetpoline())
    Features[Feature_NotUseRetpolineBit] = 1;
  return Features;
}

PredicateBitset X86InstructionSelector::
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget, const MachineFunction *MF) const {
  PredicateBitset Features;
  if (!Subtarget->isTargetWin64() ||Subtarget->getFrameLowering()->hasFP(*MF))
    Features[Feature_NotWin64WithoutFPBit] = 1;
  if (MF->getFunction().optForSize())
    Features[Feature_OptForSizeBit] = 1;
  if (MF->getFunction().optForMinSize())
    Features[Feature_OptForMinSizeBit] = 1;
  if (!MF->getFunction().optForSize())
    Features[Feature_OptForSpeedBit] = 1;
  if (!Subtarget->slowIncDec() || MF->getFunction().optForSize())
    Features[Feature_UseIncDecBit] = 1;
  return Features;
}

// LLT Objects.
enum {
  GILLT_s1,
  GILLT_s8,
  GILLT_s16,
  GILLT_s32,
  GILLT_s64,
  GILLT_s80,
  GILLT_s128,
  GILLT_v2s1,
  GILLT_v2s64,
  GILLT_v4s1,
  GILLT_v4s32,
  GILLT_v4s64,
  GILLT_v8s1,
  GILLT_v8s16,
  GILLT_v8s32,
  GILLT_v8s64,
  GILLT_v16s1,
  GILLT_v16s8,
  GILLT_v16s16,
  GILLT_v16s32,
  GILLT_v32s1,
  GILLT_v32s8,
  GILLT_v32s16,
  GILLT_v64s1,
  GILLT_v64s8,
};
const static size_t NumTypeObjects = 25;
const static LLT TypeObjects[] = {
  LLT::scalar(1),
  LLT::scalar(8),
  LLT::scalar(16),
  LLT::scalar(32),
  LLT::scalar(64),
  LLT::scalar(80),
  LLT::scalar(128),
  LLT::vector(2, 1),
  LLT::vector(2, 64),
  LLT::vector(4, 1),
  LLT::vector(4, 32),
  LLT::vector(4, 64),
  LLT::vector(8, 1),
  LLT::vector(8, 16),
  LLT::vector(8, 32),
  LLT::vector(8, 64),
  LLT::vector(16, 1),
  LLT::vector(16, 8),
  LLT::vector(16, 16),
  LLT::vector(16, 32),
  LLT::vector(32, 1),
  LLT::vector(32, 8),
  LLT::vector(32, 16),
  LLT::vector(64, 1),
  LLT::vector(64, 8),
};

// Feature bitsets.
enum {
  GIFBS_Invalid,
  GIFBS_FPStackf32,
  GIFBS_FPStackf64,
  GIFBS_Has3DNow,
  GIFBS_HasAVX,
  GIFBS_HasAVX1Only,
  GIFBS_HasAVX2,
  GIFBS_HasAVX512,
  GIFBS_HasBMI,
  GIFBS_HasBMI2,
  GIFBS_HasBWI,
  GIFBS_HasDQI,
  GIFBS_HasLWP,
  GIFBS_HasMFence,
  GIFBS_HasMMX,
  GIFBS_HasMOVBE,
  GIFBS_HasPTWRITE,
  GIFBS_HasRTM,
  GIFBS_HasSHA,
  GIFBS_HasSSE1,
  GIFBS_HasSSE2,
  GIFBS_HasSSE42,
  GIFBS_HasSSE4A,
  GIFBS_HasTBM,
  GIFBS_HasVLX,
  GIFBS_HasWAITPKG,
  GIFBS_HasWBNOINVD,
  GIFBS_HasXOP,
  GIFBS_In64BitMode,
  GIFBS_NoDQI,
  GIFBS_Not64BitMode,
  GIFBS_UseAVX,
  GIFBS_UseIncDec,
  GIFBS_UseSSE1,
  GIFBS_UseSSE2,
  GIFBS_UseSSE41,
  GIFBS_UseSSSE3,
  GIFBS_HasAES_HasAVX,
  GIFBS_HasAES_NoAVX,
  GIFBS_HasAVX_NoVLX,
  GIFBS_HasAVX_NoVLX_Or_NoBWI,
  GIFBS_HasAVX2_NoVLX,
  GIFBS_HasAVX2_NoVLX_Or_NoBWI,
  GIFBS_HasAVX512_HasVAES,
  GIFBS_HasAVX512_HasVLX,
  GIFBS_HasAVX512_HasVPCLMULQDQ,
  GIFBS_HasBWI_HasVLX,
  GIFBS_HasDQI_HasVLX,
  GIFBS_HasDQI_NoBWI,
  GIFBS_HasFSGSBase_In64BitMode,
  GIFBS_HasPCLMUL_NoAVX,
  GIFBS_HasPTWRITE_In64BitMode,
  GIFBS_HasRDPID_Not64BitMode,
  GIFBS_HasSSE2_NoAVX512,
  GIFBS_HasVAES_HasVLX,
  GIFBS_HasVAES_NoVLX,
  GIFBS_HasVLX_HasVPCLMULQDQ,
  GIFBS_HasVPCLMULQDQ_NoVLX,
  GIFBS_HasWAITPKG_In64BitMode,
  GIFBS_HasWAITPKG_Not64BitMode,
  GIFBS_Not64BitMode_OptForSize,
  GIFBS_NotWin64WithoutFP_OptForMinSize,
  GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
  GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
};
const static PredicateBitset FeatureBitsets[] {
  {}, // GIFBS_Invalid
  {Feature_FPStackf32Bit, },
  {Feature_FPStackf64Bit, },
  {Feature_Has3DNowBit, },
  {Feature_HasAVXBit, },
  {Feature_HasAVX1OnlyBit, },
  {Feature_HasAVX2Bit, },
  {Feature_HasAVX512Bit, },
  {Feature_HasBMIBit, },
  {Feature_HasBMI2Bit, },
  {Feature_HasBWIBit, },
  {Feature_HasDQIBit, },
  {Feature_HasLWPBit, },
  {Feature_HasMFenceBit, },
  {Feature_HasMMXBit, },
  {Feature_HasMOVBEBit, },
  {Feature_HasPTWRITEBit, },
  {Feature_HasRTMBit, },
  {Feature_HasSHABit, },
  {Feature_HasSSE1Bit, },
  {Feature_HasSSE2Bit, },
  {Feature_HasSSE42Bit, },
  {Feature_HasSSE4ABit, },
  {Feature_HasTBMBit, },
  {Feature_HasVLXBit, },
  {Feature_HasWAITPKGBit, },
  {Feature_HasWBNOINVDBit, },
  {Feature_HasXOPBit, },
  {Feature_In64BitModeBit, },
  {Feature_NoDQIBit, },
  {Feature_Not64BitModeBit, },
  {Feature_UseAVXBit, },
  {Feature_UseIncDecBit, },
  {Feature_UseSSE1Bit, },
  {Feature_UseSSE2Bit, },
  {Feature_UseSSE41Bit, },
  {Feature_UseSSSE3Bit, },
  {Feature_HasAESBit, Feature_HasAVXBit, },
  {Feature_HasAESBit, Feature_NoAVXBit, },
  {Feature_HasAVXBit, Feature_NoVLXBit, },
  {Feature_HasAVXBit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX2Bit, Feature_NoVLXBit, },
  {Feature_HasAVX2Bit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX512Bit, Feature_HasVAESBit, },
  {Feature_HasAVX512Bit, Feature_HasVLXBit, },
  {Feature_HasAVX512Bit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasBWIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_NoBWIBit, },
  {Feature_HasFSGSBaseBit, Feature_In64BitModeBit, },
  {Feature_HasPCLMULBit, Feature_NoAVXBit, },
  {Feature_HasPTWRITEBit, Feature_In64BitModeBit, },
  {Feature_HasRDPIDBit, Feature_Not64BitModeBit, },
  {Feature_HasSSE2Bit, Feature_NoAVX512Bit, },
  {Feature_HasVAESBit, Feature_HasVLXBit, },
  {Feature_HasVAESBit, Feature_NoVLXBit, },
  {Feature_HasVLXBit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasVPCLMULQDQBit, Feature_NoVLXBit, },
  {Feature_HasWAITPKGBit, Feature_In64BitModeBit, },
  {Feature_HasWAITPKGBit, Feature_Not64BitModeBit, },
  {Feature_Not64BitModeBit, Feature_OptForSizeBit, },
  {Feature_NotWin64WithoutFPBit, Feature_OptForMinSizeBit, },
  {Feature_HasAESBit, Feature_HasAVXBit, Feature_NoVLX_Or_NoVAESBit, },
  {Feature_HasAVXBit, Feature_HasPCLMULBit, Feature_NoVLX_Or_NoVPCLMULQDQBit, },
};

// ComplexPattern predicates.
enum {
  GICP_Invalid,
};
// See constructor for table contents

// PatFrag predicates.
enum {
  GIPFP_I64_Predicate_AndMask64 = GIPFP_I64_Invalid + 1,
  GIPFP_I64_Predicate_BTCBTSMask64,
  GIPFP_I64_Predicate_BTRMask64,
  GIPFP_I64_Predicate_FROUND_CURRENT,
  GIPFP_I64_Predicate_FROUND_NO_EXC,
  GIPFP_I64_Predicate_PrefetchWT1Level,
  GIPFP_I64_Predicate_i16immSExt8,
  GIPFP_I64_Predicate_i32immSExt8,
  GIPFP_I64_Predicate_i64immSExt32,
  GIPFP_I64_Predicate_i64immSExt8,
  GIPFP_I64_Predicate_i64immZExt32,
  GIPFP_I64_Predicate_i64immZExt32SExt8,
  GIPFP_I64_Predicate_immShift16,
  GIPFP_I64_Predicate_immShift32,
  GIPFP_I64_Predicate_immShift64,
  GIPFP_I64_Predicate_immShift8,
  GIPFP_I64_Predicate_immff00_ffff,
};
bool X86InstructionSelector::testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const {
  switch (PredicateID) {
  case GIPFP_I64_Predicate_AndMask64: {
    
  return isMask_64(Imm) && !isUInt<32>(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTCBTSMask64: {
    
  return !isInt<32>(Imm) && isPowerOf2_64(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTRMask64: {
    
  return !isUInt<32>(Imm) && !isInt<32>(Imm) && isPowerOf2_64(~Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_FROUND_CURRENT: {
    
  return Imm == X86::STATIC_ROUNDING::CUR_DIRECTION;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_FROUND_NO_EXC: {
     return Imm == 8; 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_PrefetchWT1Level: {
    
  return Imm < 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i16immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i32immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt32: {
     return isInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32: {
     return isUInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32SExt8: {
    
  return isUInt<32>(Imm) && isInt<8>(static_cast<int32_t>(Imm));

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift16: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 4;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift32: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 5;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift64: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 6;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift8: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immff00_ffff: {
    
  return Imm >= 0xff00 && Imm <= 0xffff;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
// PatFrag predicates.
enum {
  GIPFP_APFloat_Predicate_fpimm0 = GIPFP_APFloat_Invalid + 1,
  GIPFP_APFloat_Predicate_fpimm1,
  GIPFP_APFloat_Predicate_fpimmneg0,
  GIPFP_APFloat_Predicate_fpimmneg1,
};
bool X86InstructionSelector::testImmPredicate_APFloat(unsigned PredicateID, const APFloat & Imm) const {
  switch (PredicateID) {
  case GIPFP_APFloat_Predicate_fpimm0: {
    
  return Imm.isExactlyValue(+0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimm1: {
    
  return Imm.isExactlyValue(+1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg0: {
    
  return Imm.isExactlyValue(-0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg1: {
    
  return Imm.isExactlyValue(-1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
bool X86InstructionSelector::testImmPredicate_APInt(unsigned PredicateID, const APInt & Imm) const {
  llvm_unreachable("Unknown predicate");
  return false;
}

X86InstructionSelector::ComplexMatcherMemFn
X86InstructionSelector::ComplexPredicateFns[] = {
  nullptr, // GICP_Invalid
};

// Custom renderers.
enum {
  GICR_Invalid,
};
X86InstructionSelector::CustomRendererFn
X86InstructionSelector::CustomRenderers[] = {
  nullptr, // GICP_Invalid
};

bool X86InstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
  MachineFunction &MF = *I.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  // FIXME: This should be computed on a per-function basis rather than per-insn.
  AvailableFunctionFeatures = computeAvailableFunctionFeatures(&STI, &MF);
  const PredicateBitset AvailableFeatures = getAvailableFeatures();
  NewMIVector OutMIs;
  State.MIs.clear();
  State.MIs.push_back(&I);

  if (executeMatchTable(*this, OutMIs, State, ISelInfo, getMatchTable(), TII, MRI, TRI, RBI, AvailableFeatures, CoverageInfo)) {
    return true;
  }

  return false;
}

const int64_t *X86InstructionSelector::getMatchTable() const {
  constexpr static int64_t MatchTable0[] = {
    GIM_SwitchOpcode, /*MI*/0, /*[*/34, 124, /*)*//*default:*//*Label 31*/ 33353,
    /*TargetOpcode::G_ADD*//*Label 0*/ 95,
    /*TargetOpcode::G_SUB*//*Label 1*/ 1712,
    /*TargetOpcode::G_MUL*//*Label 2*/ 2793, 0, 0, 0, 0,
    /*TargetOpcode::G_AND*//*Label 3*/ 3559,
    /*TargetOpcode::G_OR*//*Label 4*/ 6764,
    /*TargetOpcode::G_XOR*//*Label 5*/ 9528, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_BITCAST*//*Label 6*/ 11594, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_INTRINSIC*//*Label 7*/ 14943,
    /*TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS*//*Label 8*/ 22750,
    /*TargetOpcode::G_ANYEXT*//*Label 9*/ 24181,
    /*TargetOpcode::G_TRUNC*//*Label 10*/ 24297,
    /*TargetOpcode::G_CONSTANT*//*Label 11*/ 24632,
    /*TargetOpcode::G_FCONSTANT*//*Label 12*/ 24835, 0, 0,
    /*TargetOpcode::G_SEXT*//*Label 13*/ 25028,
    /*TargetOpcode::G_ZEXT*//*Label 14*/ 25532,
    /*TargetOpcode::G_SHL*//*Label 15*/ 25667,
    /*TargetOpcode::G_LSHR*//*Label 16*/ 26392,
    /*TargetOpcode::G_ASHR*//*Label 17*/ 27223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FADD*//*Label 18*/ 27880,
    /*TargetOpcode::G_FSUB*//*Label 19*/ 28493,
    /*TargetOpcode::G_FMUL*//*Label 20*/ 29106, 0,
    /*TargetOpcode::G_FDIV*//*Label 21*/ 29719, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FNEG*//*Label 22*/ 30332,
    /*TargetOpcode::G_FPEXT*//*Label 23*/ 30421,
    /*TargetOpcode::G_FPTRUNC*//*Label 24*/ 30697,
    /*TargetOpcode::G_FPTOSI*//*Label 25*/ 30982,
    /*TargetOpcode::G_FPTOUI*//*Label 26*/ 31612,
    /*TargetOpcode::G_SITOFP*//*Label 27*/ 31966,
    /*TargetOpcode::G_UITOFP*//*Label 28*/ 32796, 0, 0, 0,
    /*TargetOpcode::G_BR*//*Label 29*/ 33250, 0, 0, 0,
    /*TargetOpcode::G_BSWAP*//*Label 30*/ 33263,
    // Label 0: @95
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 48*/ 1711,
    /*GILLT_s8*//*Label 32*/ 125,
    /*GILLT_s16*//*Label 33*/ 239,
    /*GILLT_s32*//*Label 34*/ 411,
    /*GILLT_s64*//*Label 35*/ 583, 0, 0, 0,
    /*GILLT_v2s64*//*Label 36*/ 783, 0,
    /*GILLT_v4s32*//*Label 37*/ 864,
    /*GILLT_v4s64*//*Label 38*/ 1067, 0,
    /*GILLT_v8s16*//*Label 39*/ 1125,
    /*GILLT_v8s32*//*Label 40*/ 1328,
    /*GILLT_v8s64*//*Label 41*/ 1386, 0,
    /*GILLT_v16s8*//*Label 42*/ 1418,
    /*GILLT_v16s16*//*Label 43*/ 1499,
    /*GILLT_v16s32*//*Label 44*/ 1557, 0,
    /*GILLT_v32s8*//*Label 45*/ 1589,
    /*GILLT_v32s16*//*Label 46*/ 1647, 0,
    /*GILLT_v64s8*//*Label 47*/ 1679,
    // Label 32: @125
    GIM_Try, /*On fail goto*//*Label 49*/ 238,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 50*/ 167, // Rule ID 15832 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, 1:{ *:[i8] })  =>  (INC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15832,
        GIR_Done,
      // Label 50: @167
      GIM_Try, /*On fail goto*//*Label 51*/ 191, // Rule ID 15836 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, -1:{ *:[i8] })  =>  (DEC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15836,
        GIR_Done,
      // Label 51: @191
      GIM_Try, /*On fail goto*//*Label 52*/ 221, // Rule ID 15784 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (ADD8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15784,
        GIR_Done,
      // Label 52: @221
      GIM_Try, /*On fail goto*//*Label 53*/ 237, // Rule ID 15778 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15778,
        GIR_Done,
      // Label 53: @237
      GIM_Reject,
    // Label 49: @238
    GIM_Reject,
    // Label 33: @239
    GIM_Try, /*On fail goto*//*Label 54*/ 410,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 55*/ 282, // Rule ID 15662 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, 128:{ *:[i16] })  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, -128:{ *:[i16] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15662,
        GIR_Done,
      // Label 55: @282
      GIM_Try, /*On fail goto*//*Label 56*/ 306, // Rule ID 15833 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, 1:{ *:[i16] })  =>  (INC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15833,
        GIR_Done,
      // Label 56: @306
      GIM_Try, /*On fail goto*//*Label 57*/ 330, // Rule ID 15837 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, -1:{ *:[i16] })  =>  (DEC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15837,
        GIR_Done,
      // Label 57: @330
      GIM_Try, /*On fail goto*//*Label 58*/ 363, // Rule ID 15787 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (ADD16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15787,
        GIR_Done,
      // Label 58: @363
      GIM_Try, /*On fail goto*//*Label 59*/ 393, // Rule ID 15785 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (ADD16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15785,
        GIR_Done,
      // Label 59: @393
      GIM_Try, /*On fail goto*//*Label 60*/ 409, // Rule ID 15779 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15779,
        GIR_Done,
      // Label 60: @409
      GIM_Reject,
    // Label 54: @410
    GIM_Reject,
    // Label 34: @411
    GIM_Try, /*On fail goto*//*Label 61*/ 582,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 62*/ 454, // Rule ID 15664 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, 128:{ *:[i32] })  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, -128:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15664,
        GIR_Done,
      // Label 62: @454
      GIM_Try, /*On fail goto*//*Label 63*/ 478, // Rule ID 15834 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] })  =>  (INC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15834,
        GIR_Done,
      // Label 63: @478
      GIM_Try, /*On fail goto*//*Label 64*/ 502, // Rule ID 15838 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] })  =>  (DEC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15838,
        GIR_Done,
      // Label 64: @502
      GIM_Try, /*On fail goto*//*Label 65*/ 535, // Rule ID 15788 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (ADD32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15788,
        GIR_Done,
      // Label 65: @535
      GIM_Try, /*On fail goto*//*Label 66*/ 565, // Rule ID 15786 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (ADD32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15786,
        GIR_Done,
      // Label 66: @565
      GIM_Try, /*On fail goto*//*Label 67*/ 581, // Rule ID 15780 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15780,
        GIR_Done,
      // Label 67: @581
      GIM_Reject,
    // Label 61: @582
    GIM_Reject,
    // Label 35: @583
    GIM_Try, /*On fail goto*//*Label 68*/ 782,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 69*/ 626, // Rule ID 15666 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 128:{ *:[i64] })  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -128:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15666,
        GIR_Done,
      // Label 69: @626
      GIM_Try, /*On fail goto*//*Label 70*/ 651, // Rule ID 15668 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 2147483648,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 2147483648:{ *:[i64] })  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -2147483648:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-2147483648,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15668,
        GIR_Done,
      // Label 70: @651
      GIM_Try, /*On fail goto*//*Label 71*/ 675, // Rule ID 15835 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] })  =>  (INC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15835,
        GIR_Done,
      // Label 71: @675
      GIM_Try, /*On fail goto*//*Label 72*/ 699, // Rule ID 15839 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] })  =>  (DEC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15839,
        GIR_Done,
      // Label 72: @699
      GIM_Try, /*On fail goto*//*Label 73*/ 732, // Rule ID 15819 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (ADD64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15819,
        GIR_Done,
      // Label 73: @732
      GIM_Try, /*On fail goto*//*Label 74*/ 765, // Rule ID 15820 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (ADD64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15820,
        GIR_Done,
      // Label 74: @765
      GIM_Try, /*On fail goto*//*Label 75*/ 781, // Rule ID 15818 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15818,
        GIR_Done,
      // Label 75: @781
      GIM_Reject,
    // Label 68: @782
    GIM_Reject,
    // Label 36: @783
    GIM_Try, /*On fail goto*//*Label 76*/ 863,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 77*/ 816, // Rule ID 1851 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1851,
        GIR_Done,
      // Label 77: @816
      GIM_Try, /*On fail goto*//*Label 78*/ 839, // Rule ID 1853 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1853,
        GIR_Done,
      // Label 78: @839
      GIM_Try, /*On fail goto*//*Label 79*/ 862, // Rule ID 3865 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPADDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3865,
        GIR_Done,
      // Label 79: @862
      GIM_Reject,
    // Label 76: @863
    GIM_Reject,
    // Label 37: @864
    GIM_Try, /*On fail goto*//*Label 80*/ 1066,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 81*/ 935, // Rule ID 12548 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2), VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12548,
        GIR_Done,
      // Label 81: @935
      GIM_Try, /*On fail goto*//*Label 82*/ 996, // Rule ID 18023 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src3, (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2))  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18023,
        GIR_Done,
      // Label 82: @996
      GIM_Try, /*On fail goto*//*Label 83*/ 1019, // Rule ID 1845 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1845,
        GIR_Done,
      // Label 83: @1019
      GIM_Try, /*On fail goto*//*Label 84*/ 1042, // Rule ID 1847 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1847,
        GIR_Done,
      // Label 84: @1042
      GIM_Try, /*On fail goto*//*Label 85*/ 1065, // Rule ID 3892 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPADDDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3892,
        GIR_Done,
      // Label 85: @1065
      GIM_Reject,
    // Label 80: @1066
    GIM_Reject,
    // Label 38: @1067
    GIM_Try, /*On fail goto*//*Label 86*/ 1124,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 87*/ 1100, // Rule ID 1855 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPADDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1855,
        GIR_Done,
      // Label 87: @1100
      GIM_Try, /*On fail goto*//*Label 88*/ 1123, // Rule ID 3856 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPADDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3856,
        GIR_Done,
      // Label 88: @1123
      GIM_Reject,
    // Label 86: @1124
    GIM_Reject,
    // Label 39: @1125
    GIM_Try, /*On fail goto*//*Label 89*/ 1327,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 90*/ 1196, // Rule ID 12547 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2), VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12547,
        GIR_Done,
      // Label 90: @1196
      GIM_Try, /*On fail goto*//*Label 91*/ 1257, // Rule ID 18022 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src3, (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2))  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18022,
        GIR_Done,
      // Label 91: @1257
      GIM_Try, /*On fail goto*//*Label 92*/ 1280, // Rule ID 1839 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1839,
        GIR_Done,
      // Label 92: @1280
      GIM_Try, /*On fail goto*//*Label 93*/ 1303, // Rule ID 1841 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1841,
        GIR_Done,
      // Label 93: @1303
      GIM_Try, /*On fail goto*//*Label 94*/ 1326, // Rule ID 3913 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPADDWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3913,
        GIR_Done,
      // Label 94: @1326
      GIM_Reject,
    // Label 89: @1327
    GIM_Reject,
    // Label 40: @1328
    GIM_Try, /*On fail goto*//*Label 95*/ 1385,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 96*/ 1361, // Rule ID 1849 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPADDDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1849,
        GIR_Done,
      // Label 96: @1361
      GIM_Try, /*On fail goto*//*Label 97*/ 1384, // Rule ID 3883 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPADDDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3883,
        GIR_Done,
      // Label 97: @1384
      GIM_Reject,
    // Label 95: @1385
    GIM_Reject,
    // Label 41: @1386
    GIM_Try, /*On fail goto*//*Label 98*/ 1417, // Rule ID 3847 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPADDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3847,
      GIR_Done,
    // Label 98: @1417
    GIM_Reject,
    // Label 42: @1418
    GIM_Try, /*On fail goto*//*Label 99*/ 1498,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 100*/ 1451, // Rule ID 1833 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1833,
        GIR_Done,
      // Label 100: @1451
      GIM_Try, /*On fail goto*//*Label 101*/ 1474, // Rule ID 1835 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1835,
        GIR_Done,
      // Label 101: @1474
      GIM_Try, /*On fail goto*//*Label 102*/ 1497, // Rule ID 3931 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPADDBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3931,
        GIR_Done,
      // Label 102: @1497
      GIM_Reject,
    // Label 99: @1498
    GIM_Reject,
    // Label 43: @1499
    GIM_Try, /*On fail goto*//*Label 103*/ 1556,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 104*/ 1532, // Rule ID 1843 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPADDWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1843,
        GIR_Done,
      // Label 104: @1532
      GIM_Try, /*On fail goto*//*Label 105*/ 1555, // Rule ID 3907 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPADDWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3907,
        GIR_Done,
      // Label 105: @1555
      GIM_Reject,
    // Label 103: @1556
    GIM_Reject,
    // Label 44: @1557
    GIM_Try, /*On fail goto*//*Label 106*/ 1588, // Rule ID 3874 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPADDDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3874,
      GIR_Done,
    // Label 106: @1588
    GIM_Reject,
    // Label 45: @1589
    GIM_Try, /*On fail goto*//*Label 107*/ 1646,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 108*/ 1622, // Rule ID 1837 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPADDBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1837,
        GIR_Done,
      // Label 108: @1622
      GIM_Try, /*On fail goto*//*Label 109*/ 1645, // Rule ID 3925 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPADDBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3925,
        GIR_Done,
      // Label 109: @1645
      GIM_Reject,
    // Label 107: @1646
    GIM_Reject,
    // Label 46: @1647
    GIM_Try, /*On fail goto*//*Label 110*/ 1678, // Rule ID 3901 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPADDWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3901,
      GIR_Done,
    // Label 110: @1678
    GIM_Reject,
    // Label 47: @1679
    GIM_Try, /*On fail goto*//*Label 111*/ 1710, // Rule ID 3919 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPADDBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3919,
      GIR_Done,
    // Label 111: @1710
    GIM_Reject,
    // Label 48: @1711
    GIM_Reject,
    // Label 1: @1712
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 128*/ 2792,
    /*GILLT_s8*//*Label 112*/ 1742,
    /*GILLT_s16*//*Label 113*/ 1808,
    /*GILLT_s32*//*Label 114*/ 1907,
    /*GILLT_s64*//*Label 115*/ 2006, 0, 0, 0,
    /*GILLT_v2s64*//*Label 116*/ 2108, 0,
    /*GILLT_v4s32*//*Label 117*/ 2189,
    /*GILLT_v4s64*//*Label 118*/ 2270, 0,
    /*GILLT_v8s16*//*Label 119*/ 2328,
    /*GILLT_v8s32*//*Label 120*/ 2409,
    /*GILLT_v8s64*//*Label 121*/ 2467, 0,
    /*GILLT_v16s8*//*Label 122*/ 2499,
    /*GILLT_v16s16*//*Label 123*/ 2580,
    /*GILLT_v16s32*//*Label 124*/ 2638, 0,
    /*GILLT_v32s8*//*Label 125*/ 2670,
    /*GILLT_v32s16*//*Label 126*/ 2728, 0,
    /*GILLT_v64s8*//*Label 127*/ 2760,
    // Label 112: @1742
    GIM_Try, /*On fail goto*//*Label 129*/ 1807,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 130*/ 1790, // Rule ID 15795 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SUB8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15795,
        GIR_Done,
      // Label 130: @1790
      GIM_Try, /*On fail goto*//*Label 131*/ 1806, // Rule ID 15789 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (SUB8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15789,
        GIR_Done,
      // Label 131: @1806
      GIM_Reject,
    // Label 129: @1807
    GIM_Reject,
    // Label 113: @1808
    GIM_Try, /*On fail goto*//*Label 132*/ 1906,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 133*/ 1859, // Rule ID 15798 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15798,
        GIR_Done,
      // Label 133: @1859
      GIM_Try, /*On fail goto*//*Label 134*/ 1889, // Rule ID 15796 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (SUB16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15796,
        GIR_Done,
      // Label 134: @1889
      GIM_Try, /*On fail goto*//*Label 135*/ 1905, // Rule ID 15790 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (SUB16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15790,
        GIR_Done,
      // Label 135: @1905
      GIM_Reject,
    // Label 132: @1906
    GIM_Reject,
    // Label 114: @1907
    GIM_Try, /*On fail goto*//*Label 136*/ 2005,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 137*/ 1958, // Rule ID 15799 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15799,
        GIR_Done,
      // Label 137: @1958
      GIM_Try, /*On fail goto*//*Label 138*/ 1988, // Rule ID 15797 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (SUB32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15797,
        GIR_Done,
      // Label 138: @1988
      GIM_Try, /*On fail goto*//*Label 139*/ 2004, // Rule ID 15791 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (SUB32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15791,
        GIR_Done,
      // Label 139: @2004
      GIM_Reject,
    // Label 136: @2005
    GIM_Reject,
    // Label 115: @2006
    GIM_Try, /*On fail goto*//*Label 140*/ 2107,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 141*/ 2057, // Rule ID 15824 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15824,
        GIR_Done,
      // Label 141: @2057
      GIM_Try, /*On fail goto*//*Label 142*/ 2090, // Rule ID 15825 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15825,
        GIR_Done,
      // Label 142: @2090
      GIM_Try, /*On fail goto*//*Label 143*/ 2106, // Rule ID 15822 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (SUB64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15822,
        GIR_Done,
      // Label 143: @2106
      GIM_Reject,
    // Label 140: @2107
    GIM_Reject,
    // Label 116: @2108
    GIM_Try, /*On fail goto*//*Label 144*/ 2188,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 145*/ 2141, // Rule ID 1917 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1917,
        GIR_Done,
      // Label 145: @2141
      GIM_Try, /*On fail goto*//*Label 146*/ 2164, // Rule ID 1919 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1919,
        GIR_Done,
      // Label 146: @2164
      GIM_Try, /*On fail goto*//*Label 147*/ 2187, // Rule ID 3955 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSUBQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3955,
        GIR_Done,
      // Label 147: @2187
      GIM_Reject,
    // Label 144: @2188
    GIM_Reject,
    // Label 117: @2189
    GIM_Try, /*On fail goto*//*Label 148*/ 2269,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 149*/ 2222, // Rule ID 1911 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1911,
        GIR_Done,
      // Label 149: @2222
      GIM_Try, /*On fail goto*//*Label 150*/ 2245, // Rule ID 1913 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1913,
        GIR_Done,
      // Label 150: @2245
      GIM_Try, /*On fail goto*//*Label 151*/ 2268, // Rule ID 3982 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSUBDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3982,
        GIR_Done,
      // Label 151: @2268
      GIM_Reject,
    // Label 148: @2269
    GIM_Reject,
    // Label 118: @2270
    GIM_Try, /*On fail goto*//*Label 152*/ 2327,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 153*/ 2303, // Rule ID 1921 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSUBQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1921,
        GIR_Done,
      // Label 153: @2303
      GIM_Try, /*On fail goto*//*Label 154*/ 2326, // Rule ID 3946 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSUBQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3946,
        GIR_Done,
      // Label 154: @2326
      GIM_Reject,
    // Label 152: @2327
    GIM_Reject,
    // Label 119: @2328
    GIM_Try, /*On fail goto*//*Label 155*/ 2408,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 156*/ 2361, // Rule ID 1905 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1905,
        GIR_Done,
      // Label 156: @2361
      GIM_Try, /*On fail goto*//*Label 157*/ 2384, // Rule ID 1907 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1907,
        GIR_Done,
      // Label 157: @2384
      GIM_Try, /*On fail goto*//*Label 158*/ 2407, // Rule ID 4003 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSUBWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4003,
        GIR_Done,
      // Label 158: @2407
      GIM_Reject,
    // Label 155: @2408
    GIM_Reject,
    // Label 120: @2409
    GIM_Try, /*On fail goto*//*Label 159*/ 2466,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 160*/ 2442, // Rule ID 1915 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSUBDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1915,
        GIR_Done,
      // Label 160: @2442
      GIM_Try, /*On fail goto*//*Label 161*/ 2465, // Rule ID 3973 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSUBDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3973,
        GIR_Done,
      // Label 161: @2465
      GIM_Reject,
    // Label 159: @2466
    GIM_Reject,
    // Label 121: @2467
    GIM_Try, /*On fail goto*//*Label 162*/ 2498, // Rule ID 3937 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSUBQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3937,
      GIR_Done,
    // Label 162: @2498
    GIM_Reject,
    // Label 122: @2499
    GIM_Try, /*On fail goto*//*Label 163*/ 2579,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 164*/ 2532, // Rule ID 1899 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1899,
        GIR_Done,
      // Label 164: @2532
      GIM_Try, /*On fail goto*//*Label 165*/ 2555, // Rule ID 1901 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1901,
        GIR_Done,
      // Label 165: @2555
      GIM_Try, /*On fail goto*//*Label 166*/ 2578, // Rule ID 4021 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPSUBBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4021,
        GIR_Done,
      // Label 166: @2578
      GIM_Reject,
    // Label 163: @2579
    GIM_Reject,
    // Label 123: @2580
    GIM_Try, /*On fail goto*//*Label 167*/ 2637,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 168*/ 2613, // Rule ID 1909 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSUBWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1909,
        GIR_Done,
      // Label 168: @2613
      GIM_Try, /*On fail goto*//*Label 169*/ 2636, // Rule ID 3997 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSUBWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3997,
        GIR_Done,
      // Label 169: @2636
      GIM_Reject,
    // Label 167: @2637
    GIM_Reject,
    // Label 124: @2638
    GIM_Try, /*On fail goto*//*Label 170*/ 2669, // Rule ID 3964 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSUBDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3964,
      GIR_Done,
    // Label 170: @2669
    GIM_Reject,
    // Label 125: @2670
    GIM_Try, /*On fail goto*//*Label 171*/ 2727,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 172*/ 2703, // Rule ID 1903 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSUBBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1903,
        GIR_Done,
      // Label 172: @2703
      GIM_Try, /*On fail goto*//*Label 173*/ 2726, // Rule ID 4015 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPSUBBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4015,
        GIR_Done,
      // Label 173: @2726
      GIM_Reject,
    // Label 171: @2727
    GIM_Reject,
    // Label 126: @2728
    GIM_Try, /*On fail goto*//*Label 174*/ 2759, // Rule ID 3991 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSUBWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3991,
      GIR_Done,
    // Label 174: @2759
    GIM_Reject,
    // Label 127: @2760
    GIM_Try, /*On fail goto*//*Label 175*/ 2791, // Rule ID 4009 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPSUBBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4009,
      GIR_Done,
    // Label 175: @2791
    GIM_Reject,
    // Label 128: @2792
    GIM_Reject,
    // Label 2: @2793
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 23, /*)*//*default:*//*Label 188*/ 3558,
    /*GILLT_s16*//*Label 176*/ 2820,
    /*GILLT_s32*//*Label 177*/ 2919,
    /*GILLT_s64*//*Label 178*/ 3018, 0, 0, 0,
    /*GILLT_v2s64*//*Label 179*/ 3120, 0,
    /*GILLT_v4s32*//*Label 180*/ 3152,
    /*GILLT_v4s64*//*Label 181*/ 3233, 0,
    /*GILLT_v8s16*//*Label 182*/ 3265,
    /*GILLT_v8s32*//*Label 183*/ 3346,
    /*GILLT_v8s64*//*Label 184*/ 3404, 0, 0,
    /*GILLT_v16s16*//*Label 185*/ 3436,
    /*GILLT_v16s32*//*Label 186*/ 3494, 0, 0,
    /*GILLT_v32s16*//*Label 187*/ 3526,
    // Label 176: @2820
    GIM_Try, /*On fail goto*//*Label 189*/ 2918,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 190*/ 2871, // Rule ID 15812 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (IMUL16rri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15812,
        GIR_Done,
      // Label 190: @2871
      GIM_Try, /*On fail goto*//*Label 191*/ 2901, // Rule ID 15810 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (IMUL16rri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15810,
        GIR_Done,
      // Label 191: @2901
      GIM_Try, /*On fail goto*//*Label 192*/ 2917, // Rule ID 15806 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (IMUL16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15806,
        GIR_Done,
      // Label 192: @2917
      GIM_Reject,
    // Label 189: @2918
    GIM_Reject,
    // Label 177: @2919
    GIM_Try, /*On fail goto*//*Label 193*/ 3017,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 194*/ 2970, // Rule ID 15813 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (IMUL32rri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15813,
        GIR_Done,
      // Label 194: @2970
      GIM_Try, /*On fail goto*//*Label 195*/ 3000, // Rule ID 15811 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (IMUL32rri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15811,
        GIR_Done,
      // Label 195: @3000
      GIM_Try, /*On fail goto*//*Label 196*/ 3016, // Rule ID 15807 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (IMUL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15807,
        GIR_Done,
      // Label 196: @3016
      GIM_Reject,
    // Label 193: @3017
    GIM_Reject,
    // Label 178: @3018
    GIM_Try, /*On fail goto*//*Label 197*/ 3119,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 198*/ 3069, // Rule ID 15828 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (IMUL64rri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15828,
        GIR_Done,
      // Label 198: @3069
      GIM_Try, /*On fail goto*//*Label 199*/ 3102, // Rule ID 15829 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (IMUL64rri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15829,
        GIR_Done,
      // Label 199: @3102
      GIM_Try, /*On fail goto*//*Label 200*/ 3118, // Rule ID 15826 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (IMUL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15826,
        GIR_Done,
      // Label 200: @3118
      GIM_Reject,
    // Label 197: @3119
    GIM_Reject,
    // Label 179: @3120
    GIM_Try, /*On fail goto*//*Label 201*/ 3151, // Rule ID 4234 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (mul:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPMULLQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4234,
      GIR_Done,
    // Label 201: @3151
    GIM_Reject,
    // Label 180: @3152
    GIM_Try, /*On fail goto*//*Label 202*/ 3232,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 203*/ 3185, // Rule ID 2539 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2539,
        GIR_Done,
      // Label 203: @3185
      GIM_Try, /*On fail goto*//*Label 204*/ 3208, // Rule ID 2547 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2547,
        GIR_Done,
      // Label 204: @3208
      GIM_Try, /*On fail goto*//*Label 205*/ 3231, // Rule ID 4189 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPMULLDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4189,
        GIR_Done,
      // Label 205: @3231
      GIM_Reject,
    // Label 202: @3232
    GIM_Reject,
    // Label 181: @3233
    GIM_Try, /*On fail goto*//*Label 206*/ 3264, // Rule ID 4225 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (mul:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPMULLQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4225,
      GIR_Done,
    // Label 206: @3264
    GIM_Reject,
    // Label 182: @3265
    GIM_Try, /*On fail goto*//*Label 207*/ 3345,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 208*/ 3298, // Rule ID 1881 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1881,
        GIR_Done,
      // Label 208: @3298
      GIM_Try, /*On fail goto*//*Label 209*/ 3321, // Rule ID 1883 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1883,
        GIR_Done,
      // Label 209: @3321
      GIM_Try, /*On fail goto*//*Label 210*/ 3344, // Rule ID 4210 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPMULLWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4210,
        GIR_Done,
      // Label 210: @3344
      GIM_Reject,
    // Label 207: @3345
    GIM_Reject,
    // Label 183: @3346
    GIM_Try, /*On fail goto*//*Label 211*/ 3403,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 212*/ 3379, // Rule ID 2543 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPMULLDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2543,
        GIR_Done,
      // Label 212: @3379
      GIM_Try, /*On fail goto*//*Label 213*/ 3402, // Rule ID 4180 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPMULLDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4180,
        GIR_Done,
      // Label 213: @3402
      GIM_Reject,
    // Label 211: @3403
    GIM_Reject,
    // Label 184: @3404
    GIM_Try, /*On fail goto*//*Label 214*/ 3435, // Rule ID 4216 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPMULLQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4216,
      GIR_Done,
    // Label 214: @3435
    GIM_Reject,
    // Label 185: @3436
    GIM_Try, /*On fail goto*//*Label 215*/ 3493,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 216*/ 3469, // Rule ID 1885 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPMULLWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1885,
        GIR_Done,
      // Label 216: @3469
      GIM_Try, /*On fail goto*//*Label 217*/ 3492, // Rule ID 4204 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPMULLWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4204,
        GIR_Done,
      // Label 217: @3492
      GIM_Reject,
    // Label 215: @3493
    GIM_Reject,
    // Label 186: @3494
    GIM_Try, /*On fail goto*//*Label 218*/ 3525, // Rule ID 4171 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPMULLDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4171,
      GIR_Done,
    // Label 218: @3525
    GIM_Reject,
    // Label 187: @3526
    GIM_Try, /*On fail goto*//*Label 219*/ 3557, // Rule ID 4198 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPMULLWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4198,
      GIR_Done,
    // Label 219: @3557
    GIM_Reject,
    // Label 188: @3558
    GIM_Reject,
    // Label 3: @3559
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 235*/ 6763,
    /*GILLT_s1*//*Label 220*/ 3589,
    /*GILLT_s8*//*Label 221*/ 3887,
    /*GILLT_s16*//*Label 222*/ 3953,
    /*GILLT_s32*//*Label 223*/ 4052,
    /*GILLT_s64*//*Label 224*/ 5213, 0,
    /*GILLT_s128*//*Label 225*/ 6085,
    /*GILLT_v2s1*//*Label 226*/ 6178,
    /*GILLT_v2s64*//*Label 227*/ 6271,
    /*GILLT_v4s1*//*Label 228*/ 6352, 0,
    /*GILLT_v4s64*//*Label 229*/ 6445,
    /*GILLT_v8s1*//*Label 230*/ 6526, 0, 0,
    /*GILLT_v8s64*//*Label 231*/ 6635,
    /*GILLT_v16s1*//*Label 232*/ 6667, 0, 0, 0,
    /*GILLT_v32s1*//*Label 233*/ 6699, 0, 0,
    /*GILLT_v64s1*//*Label 234*/ 6731,
    // Label 220: @3589
    GIM_Try, /*On fail goto*//*Label 236*/ 3886,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 237*/ 3704, // Rule ID 13907 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 13907,
        GIR_Done,
      // Label 237: @3704
      GIM_Try, /*On fail goto*//*Label 238*/ 3805, // Rule ID 18106 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 18106,
        GIR_Done,
      // Label 238: @3805
      GIM_Try, /*On fail goto*//*Label 239*/ 3885, // Rule ID 13903 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 13903,
        GIR_Done,
      // Label 239: @3885
      GIM_Reject,
    // Label 236: @3886
    GIM_Reject,
    // Label 221: @3887
    GIM_Try, /*On fail goto*//*Label 240*/ 3952,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 241*/ 3935, // Rule ID 15878 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AND8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15878,
        GIR_Done,
      // Label 241: @3935
      GIM_Try, /*On fail goto*//*Label 242*/ 3951, // Rule ID 15870 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (AND8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15870,
        GIR_Done,
      // Label 242: @3951
      GIM_Reject,
    // Label 240: @3952
    GIM_Reject,
    // Label 222: @3953
    GIM_Try, /*On fail goto*//*Label 243*/ 4051,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 244*/ 4004, // Rule ID 15881 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (AND16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15881,
        GIR_Done,
      // Label 244: @4004
      GIM_Try, /*On fail goto*//*Label 245*/ 4034, // Rule ID 15879 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (AND16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15879,
        GIR_Done,
      // Label 245: @4034
      GIM_Try, /*On fail goto*//*Label 246*/ 4050, // Rule ID 15871 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (AND16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15871,
        GIR_Done,
      // Label 246: @4050
      GIM_Reject,
    // Label 243: @4051
    GIM_Reject,
    // Label 223: @4052
    GIM_Try, /*On fail goto*//*Label 247*/ 5212,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 248*/ 4153, // Rule ID 18400 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_LSHR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        // (and:{ *:[i32] } (srl:{ *:[i32] } -1:{ *:[i32] }, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))), GR32:{ *:[i32] }:$src)  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18400,
        GIR_Done,
      // Label 248: @4153
      GIM_Try, /*On fail goto*//*Label 249*/ 4248, // Rule ID 16213 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_LSHR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (srl:{ *:[i32] } -1:{ *:[i32] }, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))))  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16213,
        GIR_Done,
      // Label 249: @4248
      GIM_Try, /*On fail goto*//*Label 250*/ 4323, // Rule ID 17996 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17996,
        GIR_Done,
      // Label 250: @4323
      GIM_Try, /*On fail goto*//*Label 251*/ 4398, // Rule ID 18008 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18008,
        GIR_Done,
      // Label 251: @4398
      GIM_Try, /*On fail goto*//*Label 252*/ 4473, // Rule ID 12451 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12451,
        GIR_Done,
      // Label 252: @4473
      GIM_Try, /*On fail goto*//*Label 253*/ 4548, // Rule ID 12463 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12463,
        GIR_Done,
      // Label 253: @4548
      GIM_Try, /*On fail goto*//*Label 254*/ 4602, // Rule ID 17990 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17990,
        GIR_Done,
      // Label 254: @4602
      GIM_Try, /*On fail goto*//*Label 255*/ 4656, // Rule ID 18382 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18382,
        GIR_Done,
      // Label 255: @4656
      GIM_Try, /*On fail goto*//*Label 256*/ 4710, // Rule ID 18386 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18386,
        GIR_Done,
      // Label 256: @4710
      GIM_Try, /*On fail goto*//*Label 257*/ 4764, // Rule ID 12445 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12445,
        GIR_Done,
      // Label 257: @4764
      GIM_Try, /*On fail goto*//*Label 258*/ 4818, // Rule ID 16187 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16187,
        GIR_Done,
      // Label 258: @4818
      GIM_Try, /*On fail goto*//*Label 259*/ 4872, // Rule ID 16191 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16191,
        GIR_Done,
      // Label 259: @4872
      GIM_Try, /*On fail goto*//*Label 260*/ 4923, // Rule ID 15672 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 65535,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 65535:{ *:[i32] })  =>  (MOVZX32rr16:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src1, sub_16bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/3, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15672,
        GIR_Done,
      // Label 260: @4923
      GIM_Try, /*On fail goto*//*Label 261*/ 4974, // Rule ID 15673 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 255,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 255:{ *:[i32] })  =>  (MOVZX32rr8:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src1, sub_8bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s8,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15673,
        GIR_Done,
      // Label 261: @4974
      GIM_Try, /*On fail goto*//*Label 262*/ 5019, // Rule ID 15882 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (AND32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15882,
        GIR_Done,
      // Label 262: @5019
      GIM_Try, /*On fail goto*//*Label 263*/ 5061, // Rule ID 15880 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (AND32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15880,
        GIR_Done,
      // Label 263: @5061
      GIM_Try, /*On fail goto*//*Label 264*/ 5122, // Rule ID 12465 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src2)  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12465,
        GIR_Done,
      // Label 264: @5122
      GIM_Try, /*On fail goto*//*Label 265*/ 5183, // Rule ID 18010 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src2, (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }))  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18010,
        GIR_Done,
      // Label 265: @5183
      GIM_Try, /*On fail goto*//*Label 266*/ 5211, // Rule ID 15872 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (AND32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15872,
        GIR_Done,
      // Label 266: @5211
      GIM_Reject,
    // Label 247: @5212
    GIM_Reject,
    // Label 224: @5213
    GIM_Try, /*On fail goto*//*Label 267*/ 6084,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 268*/ 5294, // Rule ID 17997 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17997,
        GIR_Done,
      // Label 268: @5294
      GIM_Try, /*On fail goto*//*Label 269*/ 5369, // Rule ID 18009 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18009,
        GIR_Done,
      // Label 269: @5369
      GIM_Try, /*On fail goto*//*Label 270*/ 5444, // Rule ID 12452 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12452,
        GIR_Done,
      // Label 270: @5444
      GIM_Try, /*On fail goto*//*Label 271*/ 5519, // Rule ID 12464 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12464,
        GIR_Done,
      // Label 271: @5519
      GIM_Try, /*On fail goto*//*Label 272*/ 5573, // Rule ID 17991 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17991,
        GIR_Done,
      // Label 272: @5573
      GIM_Try, /*On fail goto*//*Label 273*/ 5627, // Rule ID 18383 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18383,
        GIR_Done,
      // Label 273: @5627
      GIM_Try, /*On fail goto*//*Label 274*/ 5681, // Rule ID 18387 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18387,
        GIR_Done,
      // Label 274: @5681
      GIM_Try, /*On fail goto*//*Label 275*/ 5735, // Rule ID 12446 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12446,
        GIR_Done,
      // Label 275: @5735
      GIM_Try, /*On fail goto*//*Label 276*/ 5789, // Rule ID 16188 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16188,
        GIR_Done,
      // Label 276: @5789
      GIM_Try, /*On fail goto*//*Label 277*/ 5843, // Rule ID 16192 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16192,
        GIR_Done,
      // Label 277: @5843
      GIM_Try, /*On fail goto*//*Label 278*/ 5888, // Rule ID 15883 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (AND64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15883,
        GIR_Done,
      // Label 278: @5888
      GIM_Try, /*On fail goto*//*Label 279*/ 5933, // Rule ID 15884 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (AND64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15884,
        GIR_Done,
      // Label 279: @5933
      GIM_Try, /*On fail goto*//*Label 280*/ 5994, // Rule ID 12466 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src2)  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12466,
        GIR_Done,
      // Label 280: @5994
      GIM_Try, /*On fail goto*//*Label 281*/ 6055, // Rule ID 18011 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src2, (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }))  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18011,
        GIR_Done,
      // Label 281: @6055
      GIM_Try, /*On fail goto*//*Label 282*/ 6083, // Rule ID 15873 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (AND64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15873,
        GIR_Done,
      // Label 282: @6083
      GIM_Reject,
    // Label 267: @6084
    GIM_Reject,
    // Label 225: @6085
    GIM_Try, /*On fail goto*//*Label 283*/ 6177, // Rule ID 13382 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s128,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s128,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR128RegClassID,
      // (and:{ *:[i128] } FR128:{ *:[i128] }:$src1, FR128:{ *:[i128] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[i128] } (ANDPSrr:{ *:[v16i8] } (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src1, VR128:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src2, VR128:{ *:[i32] })), FR128:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s8,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::ANDPSrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC FR128*/73,
      // GIR_Coverage, 13382,
      GIR_Done,
    // Label 283: @6177
    GIM_Reject,
    // Label 226: @6178
    GIM_Try, /*On fail goto*//*Label 284*/ 6270, // Rule ID 13904 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (and:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13904,
      GIR_Done,
    // Label 284: @6270
    GIM_Reject,
    // Label 227: @6271
    GIM_Try, /*On fail goto*//*Label 285*/ 6351,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 286*/ 6304, // Rule ID 1573 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1573,
        GIR_Done,
      // Label 286: @6304
      GIM_Try, /*On fail goto*//*Label 287*/ 6327, // Rule ID 1575 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1575,
        GIR_Done,
      // Label 287: @6327
      GIM_Try, /*On fail goto*//*Label 288*/ 6350, // Rule ID 4918 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPANDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4918,
        GIR_Done,
      // Label 288: @6350
      GIM_Reject,
    // Label 285: @6351
    GIM_Reject,
    // Label 228: @6352
    GIM_Try, /*On fail goto*//*Label 289*/ 6444, // Rule ID 13905 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (and:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13905,
      GIR_Done,
    // Label 289: @6444
    GIM_Reject,
    // Label 229: @6445
    GIM_Try, /*On fail goto*//*Label 290*/ 6525,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 291*/ 6478, // Rule ID 1577 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPANDYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1577,
        GIR_Done,
      // Label 291: @6478
      GIM_Try, /*On fail goto*//*Label 292*/ 6501, // Rule ID 4909 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPANDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4909,
        GIR_Done,
      // Label 292: @6501
      GIM_Try, /*On fail goto*//*Label 293*/ 6524, // Rule ID 12715 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VANDPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12715,
        GIR_Done,
      // Label 293: @6524
      GIM_Reject,
    // Label 290: @6525
    GIM_Reject,
    // Label 230: @6526
    GIM_Try, /*On fail goto*//*Label 294*/ 6634,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 295*/ 6559, // Rule ID 3633 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KANDBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3633,
        GIR_Done,
      // Label 295: @6559
      GIM_Try, /*On fail goto*//*Label 296*/ 6633, // Rule ID 13902 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/11,
        // GIR_Coverage, 13902,
        GIR_Done,
      // Label 296: @6633
      GIM_Reject,
    // Label 294: @6634
    GIM_Reject,
    // Label 231: @6635
    GIM_Try, /*On fail goto*//*Label 297*/ 6666, // Rule ID 4900 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPANDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4900,
      GIR_Done,
    // Label 297: @6666
    GIM_Reject,
    // Label 232: @6667
    GIM_Try, /*On fail goto*//*Label 298*/ 6698, // Rule ID 3634 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (and:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KANDWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3634,
      GIR_Done,
    // Label 298: @6698
    GIM_Reject,
    // Label 233: @6699
    GIM_Try, /*On fail goto*//*Label 299*/ 6730, // Rule ID 3635 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (and:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KANDDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3635,
      GIR_Done,
    // Label 299: @6730
    GIM_Reject,
    // Label 234: @6731
    GIM_Try, /*On fail goto*//*Label 300*/ 6762, // Rule ID 3636 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (and:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KANDQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3636,
      GIR_Done,
    // Label 300: @6762
    GIM_Reject,
    // Label 235: @6763
    GIM_Reject,
    // Label 4: @6764
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 316*/ 9527,
    /*GILLT_s1*//*Label 301*/ 6794,
    /*GILLT_s8*//*Label 302*/ 6887,
    /*GILLT_s16*//*Label 303*/ 6953,
    /*GILLT_s32*//*Label 304*/ 7052,
    /*GILLT_s64*//*Label 305*/ 7949, 0,
    /*GILLT_s128*//*Label 306*/ 8849,
    /*GILLT_v2s1*//*Label 307*/ 8942,
    /*GILLT_v2s64*//*Label 308*/ 9035,
    /*GILLT_v4s1*//*Label 309*/ 9116, 0,
    /*GILLT_v4s64*//*Label 310*/ 9209,
    /*GILLT_v8s1*//*Label 311*/ 9290, 0, 0,
    /*GILLT_v8s64*//*Label 312*/ 9399,
    /*GILLT_v16s1*//*Label 313*/ 9431, 0, 0, 0,
    /*GILLT_v32s1*//*Label 314*/ 9463, 0, 0,
    /*GILLT_v64s1*//*Label 315*/ 9495,
    // Label 301: @6794
    GIM_Try, /*On fail goto*//*Label 317*/ 6886, // Rule ID 13911 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
      // (or:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13911,
      GIR_Done,
    // Label 317: @6886
    GIM_Reject,
    // Label 302: @6887
    GIM_Try, /*On fail goto*//*Label 318*/ 6952,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 319*/ 6935, // Rule ID 15848 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (OR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15848,
        GIR_Done,
      // Label 319: @6935
      GIM_Try, /*On fail goto*//*Label 320*/ 6951, // Rule ID 15840 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (OR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15840,
        GIR_Done,
      // Label 320: @6951
      GIM_Reject,
    // Label 318: @6952
    GIM_Reject,
    // Label 303: @6953
    GIM_Try, /*On fail goto*//*Label 321*/ 7051,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 322*/ 7004, // Rule ID 15851 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (OR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15851,
        GIR_Done,
      // Label 322: @7004
      GIM_Try, /*On fail goto*//*Label 323*/ 7034, // Rule ID 15849 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (OR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15849,
        GIR_Done,
      // Label 323: @7034
      GIM_Try, /*On fail goto*//*Label 324*/ 7050, // Rule ID 15841 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (OR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15841,
        GIR_Done,
      // Label 324: @7050
      GIM_Reject,
    // Label 321: @7051
    GIM_Reject,
    // Label 304: @7052
    GIM_Try, /*On fail goto*//*Label 325*/ 7948,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 326*/ 7133, // Rule ID 18004 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18004,
        GIR_Done,
      // Label 326: @7133
      GIM_Try, /*On fail goto*//*Label 327*/ 7208, // Rule ID 18006 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18006,
        GIR_Done,
      // Label 327: @7208
      GIM_Try, /*On fail goto*//*Label 328*/ 7283, // Rule ID 17992 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17992,
        GIR_Done,
      // Label 328: @7283
      GIM_Try, /*On fail goto*//*Label 329*/ 7358, // Rule ID 12459 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12459,
        GIR_Done,
      // Label 329: @7358
      GIM_Try, /*On fail goto*//*Label 330*/ 7433, // Rule ID 12461 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12461,
        GIR_Done,
      // Label 330: @7433
      GIM_Try, /*On fail goto*//*Label 331*/ 7508, // Rule ID 12447 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12447,
        GIR_Done,
      // Label 331: @7508
      GIM_Try, /*On fail goto*//*Label 332*/ 7562, // Rule ID 18000 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18000,
        GIR_Done,
      // Label 332: @7562
      GIM_Try, /*On fail goto*//*Label 333*/ 7616, // Rule ID 18002 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18002,
        GIR_Done,
      // Label 333: @7616
      GIM_Try, /*On fail goto*//*Label 334*/ 7670, // Rule ID 17994 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17994,
        GIR_Done,
      // Label 334: @7670
      GIM_Try, /*On fail goto*//*Label 335*/ 7724, // Rule ID 12455 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12455,
        GIR_Done,
      // Label 335: @7724
      GIM_Try, /*On fail goto*//*Label 336*/ 7778, // Rule ID 12457 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12457,
        GIR_Done,
      // Label 336: @7778
      GIM_Try, /*On fail goto*//*Label 337*/ 7832, // Rule ID 12449 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12449,
        GIR_Done,
      // Label 337: @7832
      GIM_Try, /*On fail goto*//*Label 338*/ 7877, // Rule ID 15852 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (OR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15852,
        GIR_Done,
      // Label 338: @7877
      GIM_Try, /*On fail goto*//*Label 339*/ 7919, // Rule ID 15850 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (OR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15850,
        GIR_Done,
      // Label 339: @7919
      GIM_Try, /*On fail goto*//*Label 340*/ 7947, // Rule ID 15842 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (OR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15842,
        GIR_Done,
      // Label 340: @7947
      GIM_Reject,
    // Label 325: @7948
    GIM_Reject,
    // Label 305: @7949
    GIM_Try, /*On fail goto*//*Label 341*/ 8848,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 342*/ 8030, // Rule ID 18005 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18005,
        GIR_Done,
      // Label 342: @8030
      GIM_Try, /*On fail goto*//*Label 343*/ 8105, // Rule ID 18007 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18007,
        GIR_Done,
      // Label 343: @8105
      GIM_Try, /*On fail goto*//*Label 344*/ 8180, // Rule ID 17993 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17993,
        GIR_Done,
      // Label 344: @8180
      GIM_Try, /*On fail goto*//*Label 345*/ 8255, // Rule ID 12460 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12460,
        GIR_Done,
      // Label 345: @8255
      GIM_Try, /*On fail goto*//*Label 346*/ 8330, // Rule ID 12462 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12462,
        GIR_Done,
      // Label 346: @8330
      GIM_Try, /*On fail goto*//*Label 347*/ 8405, // Rule ID 12448 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12448,
        GIR_Done,
      // Label 347: @8405
      GIM_Try, /*On fail goto*//*Label 348*/ 8459, // Rule ID 18001 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18001,
        GIR_Done,
      // Label 348: @8459
      GIM_Try, /*On fail goto*//*Label 349*/ 8513, // Rule ID 18003 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18003,
        GIR_Done,
      // Label 349: @8513
      GIM_Try, /*On fail goto*//*Label 350*/ 8567, // Rule ID 17995 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17995,
        GIR_Done,
      // Label 350: @8567
      GIM_Try, /*On fail goto*//*Label 351*/ 8621, // Rule ID 12456 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12456,
        GIR_Done,
      // Label 351: @8621
      GIM_Try, /*On fail goto*//*Label 352*/ 8675, // Rule ID 12458 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12458,
        GIR_Done,
      // Label 352: @8675
      GIM_Try, /*On fail goto*//*Label 353*/ 8729, // Rule ID 12450 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12450,
        GIR_Done,
      // Label 353: @8729
      GIM_Try, /*On fail goto*//*Label 354*/ 8774, // Rule ID 15853 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (OR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15853,
        GIR_Done,
      // Label 354: @8774
      GIM_Try, /*On fail goto*//*Label 355*/ 8819, // Rule ID 15854 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (OR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15854,
        GIR_Done,
      // Label 355: @8819
      GIM_Try, /*On fail goto*//*Label 356*/ 8847, // Rule ID 15843 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (OR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15843,
        GIR_Done,
      // Label 356: @8847
      GIM_Reject,
    // Label 341: @8848
    GIM_Reject,
    // Label 306: @8849
    GIM_Try, /*On fail goto*//*Label 357*/ 8941, // Rule ID 13385 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s128,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s128,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR128RegClassID,
      // (or:{ *:[i128] } FR128:{ *:[i128] }:$src1, FR128:{ *:[i128] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[i128] } (ORPSrr:{ *:[v16i8] } (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src1, VR128:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src2, VR128:{ *:[i32] })), FR128:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s8,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::ORPSrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC FR128*/73,
      // GIR_Coverage, 13385,
      GIR_Done,
    // Label 357: @8941
    GIM_Reject,
    // Label 307: @8942
    GIM_Try, /*On fail goto*//*Label 358*/ 9034, // Rule ID 13912 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (or:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13912,
      GIR_Done,
    // Label 358: @9034
    GIM_Reject,
    // Label 308: @9035
    GIM_Try, /*On fail goto*//*Label 359*/ 9115,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 360*/ 9068, // Rule ID 1579 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1579,
        GIR_Done,
      // Label 360: @9068
      GIM_Try, /*On fail goto*//*Label 361*/ 9091, // Rule ID 1581 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1581,
        GIR_Done,
      // Label 361: @9091
      GIM_Try, /*On fail goto*//*Label 362*/ 9114, // Rule ID 4966 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4966,
        GIR_Done,
      // Label 362: @9114
      GIM_Reject,
    // Label 359: @9115
    GIM_Reject,
    // Label 309: @9116
    GIM_Try, /*On fail goto*//*Label 363*/ 9208, // Rule ID 13913 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (or:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13913,
      GIR_Done,
    // Label 363: @9208
    GIM_Reject,
    // Label 310: @9209
    GIM_Try, /*On fail goto*//*Label 364*/ 9289,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 365*/ 9242, // Rule ID 1583 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1583,
        GIR_Done,
      // Label 365: @9242
      GIM_Try, /*On fail goto*//*Label 366*/ 9265, // Rule ID 4957 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4957,
        GIR_Done,
      // Label 366: @9265
      GIM_Try, /*On fail goto*//*Label 367*/ 9288, // Rule ID 12716 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12716,
        GIR_Done,
      // Label 367: @9288
      GIM_Reject,
    // Label 364: @9289
    GIM_Reject,
    // Label 311: @9290
    GIM_Try, /*On fail goto*//*Label 368*/ 9398,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 369*/ 9323, // Rule ID 3637 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3637,
        GIR_Done,
      // Label 369: @9323
      GIM_Try, /*On fail goto*//*Label 370*/ 9397, // Rule ID 13910 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/11,
        // GIR_Coverage, 13910,
        GIR_Done,
      // Label 370: @9397
      GIM_Reject,
    // Label 368: @9398
    GIM_Reject,
    // Label 312: @9399
    GIM_Try, /*On fail goto*//*Label 371*/ 9430, // Rule ID 4948 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4948,
      GIR_Done,
    // Label 371: @9430
    GIM_Reject,
    // Label 313: @9431
    GIM_Try, /*On fail goto*//*Label 372*/ 9462, // Rule ID 3638 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (or:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3638,
      GIR_Done,
    // Label 372: @9462
    GIM_Reject,
    // Label 314: @9463
    GIM_Try, /*On fail goto*//*Label 373*/ 9494, // Rule ID 3639 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (or:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3639,
      GIR_Done,
    // Label 373: @9494
    GIM_Reject,
    // Label 315: @9495
    GIM_Try, /*On fail goto*//*Label 374*/ 9526, // Rule ID 3640 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (or:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3640,
      GIR_Done,
    // Label 374: @9526
    GIM_Reject,
    // Label 316: @9527
    GIM_Reject,
    // Label 5: @9528
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 390*/ 11593,
    /*GILLT_s1*//*Label 375*/ 9558,
    /*GILLT_s8*//*Label 376*/ 9957,
    /*GILLT_s16*//*Label 377*/ 10045,
    /*GILLT_s32*//*Label 378*/ 10166,
    /*GILLT_s64*//*Label 379*/ 10539, 0,
    /*GILLT_s128*//*Label 380*/ 10915,
    /*GILLT_v2s1*//*Label 381*/ 11008,
    /*GILLT_v2s64*//*Label 382*/ 11101,
    /*GILLT_v4s1*//*Label 383*/ 11182, 0,
    /*GILLT_v4s64*//*Label 384*/ 11275,
    /*GILLT_v8s1*//*Label 385*/ 11356, 0, 0,
    /*GILLT_v8s64*//*Label 386*/ 11465,
    /*GILLT_v16s1*//*Label 387*/ 11497, 0, 0, 0,
    /*GILLT_v32s1*//*Label 388*/ 11529, 0, 0,
    /*GILLT_v64s1*//*Label 389*/ 11561,
    // Label 375: @9558
    GIM_Try, /*On fail goto*//*Label 391*/ 9956,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 392*/ 9673, // Rule ID 18118 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 18118,
        GIR_Done,
      // Label 392: @9673
      GIM_Try, /*On fail goto*//*Label 393*/ 9774, // Rule ID 13915 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2), -1:{ *:[v1i1] })  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 13915,
        GIR_Done,
      // Label 393: @9774
      GIM_Try, /*On fail goto*//*Label 394*/ 9875, // Rule ID 18119 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 18119,
        GIR_Done,
      // Label 394: @9875
      GIM_Try, /*On fail goto*//*Label 395*/ 9955, // Rule ID 13919 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
        // GIR_Coverage, 13919,
        GIR_Done,
      // Label 395: @9955
      GIM_Reject,
    // Label 391: @9956
    GIM_Reject,
    // Label 376: @9957
    GIM_Try, /*On fail goto*//*Label 396*/ 10044,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 397*/ 9997, // Rule ID 152 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, -1:{ *:[i8] })  =>  (NOT8r:{ *:[i8] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 152,
        GIR_Done,
      // Label 397: @9997
      GIM_Try, /*On fail goto*//*Label 398*/ 10027, // Rule ID 15863 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (XOR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15863,
        GIR_Done,
      // Label 398: @10027
      GIM_Try, /*On fail goto*//*Label 399*/ 10043, // Rule ID 15855 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (XOR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15855,
        GIR_Done,
      // Label 399: @10043
      GIM_Reject,
    // Label 396: @10044
    GIM_Reject,
    // Label 377: @10045
    GIM_Try, /*On fail goto*//*Label 400*/ 10165,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 401*/ 10085, // Rule ID 153 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, -1:{ *:[i16] })  =>  (NOT16r:{ *:[i16] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 153,
        GIR_Done,
      // Label 401: @10085
      GIM_Try, /*On fail goto*//*Label 402*/ 10118, // Rule ID 15866 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (XOR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15866,
        GIR_Done,
      // Label 402: @10118
      GIM_Try, /*On fail goto*//*Label 403*/ 10148, // Rule ID 15864 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (XOR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15864,
        GIR_Done,
      // Label 403: @10148
      GIM_Try, /*On fail goto*//*Label 404*/ 10164, // Rule ID 15856 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (XOR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15856,
        GIR_Done,
      // Label 404: @10164
      GIM_Reject,
    // Label 400: @10165
    GIM_Reject,
    // Label 378: @10166
    GIM_Try, /*On fail goto*//*Label 405*/ 10538,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 406*/ 10206, // Rule ID 154 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] })  =>  (NOT32r:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 154,
        GIR_Done,
      // Label 406: @10206
      GIM_Try, /*On fail goto*//*Label 407*/ 10260, // Rule ID 17998 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17998,
        GIR_Done,
      // Label 407: @10260
      GIM_Try, /*On fail goto*//*Label 408*/ 10314, // Rule ID 18384 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18384,
        GIR_Done,
      // Label 408: @10314
      GIM_Try, /*On fail goto*//*Label 409*/ 10368, // Rule ID 12453 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12453,
        GIR_Done,
      // Label 409: @10368
      GIM_Try, /*On fail goto*//*Label 410*/ 10422, // Rule ID 16189 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16189,
        GIR_Done,
      // Label 410: @10422
      GIM_Try, /*On fail goto*//*Label 411*/ 10467, // Rule ID 15867 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (XOR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15867,
        GIR_Done,
      // Label 411: @10467
      GIM_Try, /*On fail goto*//*Label 412*/ 10509, // Rule ID 15865 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (XOR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15865,
        GIR_Done,
      // Label 412: @10509
      GIM_Try, /*On fail goto*//*Label 413*/ 10537, // Rule ID 15857 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (XOR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15857,
        GIR_Done,
      // Label 413: @10537
      GIM_Reject,
    // Label 405: @10538
    GIM_Reject,
    // Label 379: @10539
    GIM_Try, /*On fail goto*//*Label 414*/ 10914,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 415*/ 10579, // Rule ID 155 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] })  =>  (NOT64r:{ *:[i64] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 155,
        GIR_Done,
      // Label 415: @10579
      GIM_Try, /*On fail goto*//*Label 416*/ 10633, // Rule ID 17999 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17999,
        GIR_Done,
      // Label 416: @10633
      GIM_Try, /*On fail goto*//*Label 417*/ 10687, // Rule ID 18385 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18385,
        GIR_Done,
      // Label 417: @10687
      GIM_Try, /*On fail goto*//*Label 418*/ 10741, // Rule ID 12454 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12454,
        GIR_Done,
      // Label 418: @10741
      GIM_Try, /*On fail goto*//*Label 419*/ 10795, // Rule ID 16190 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16190,
        GIR_Done,
      // Label 419: @10795
      GIM_Try, /*On fail goto*//*Label 420*/ 10840, // Rule ID 15868 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (XOR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15868,
        GIR_Done,
      // Label 420: @10840
      GIM_Try, /*On fail goto*//*Label 421*/ 10885, // Rule ID 15869 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (XOR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15869,
        GIR_Done,
      // Label 421: @10885
      GIM_Try, /*On fail goto*//*Label 422*/ 10913, // Rule ID 15858 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (XOR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15858,
        GIR_Done,
      // Label 422: @10913
      GIM_Reject,
    // Label 414: @10914
    GIM_Reject,
    // Label 380: @10915
    GIM_Try, /*On fail goto*//*Label 423*/ 11007, // Rule ID 13388 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s128,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s128,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR128RegClassID,
      // (xor:{ *:[i128] } FR128:{ *:[i128] }:$src1, FR128:{ *:[i128] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[i128] } (XORPSrr:{ *:[v16i8] } (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src1, VR128:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i8] } FR128:{ *:[i128] }:$src2, VR128:{ *:[i32] })), FR128:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s8,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s8,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::XORPSrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC FR128*/73,
      // GIR_Coverage, 13388,
      GIR_Done,
    // Label 423: @11007
    GIM_Reject,
    // Label 381: @11008
    GIM_Try, /*On fail goto*//*Label 424*/ 11100, // Rule ID 13920 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (xor:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13920,
      GIR_Done,
    // Label 424: @11100
    GIM_Reject,
    // Label 382: @11101
    GIM_Try, /*On fail goto*//*Label 425*/ 11181,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 426*/ 11134, // Rule ID 1585 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1585,
        GIR_Done,
      // Label 426: @11134
      GIM_Try, /*On fail goto*//*Label 427*/ 11157, // Rule ID 1587 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1587,
        GIR_Done,
      // Label 427: @11157
      GIM_Try, /*On fail goto*//*Label 428*/ 11180, // Rule ID 5014 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPXORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5014,
        GIR_Done,
      // Label 428: @11180
      GIM_Reject,
    // Label 425: @11181
    GIM_Reject,
    // Label 383: @11182
    GIM_Try, /*On fail goto*//*Label 429*/ 11274, // Rule ID 13921 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (xor:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/7,
      // GIR_Coverage, 13921,
      GIR_Done,
    // Label 429: @11274
    GIM_Reject,
    // Label 384: @11275
    GIM_Try, /*On fail goto*//*Label 430*/ 11355,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 431*/ 11308, // Rule ID 1589 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPXORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1589,
        GIR_Done,
      // Label 431: @11308
      GIM_Try, /*On fail goto*//*Label 432*/ 11331, // Rule ID 5005 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPXORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5005,
        GIR_Done,
      // Label 432: @11331
      GIM_Try, /*On fail goto*//*Label 433*/ 11354, // Rule ID 12717 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VXORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12717,
        GIR_Done,
      // Label 433: @11354
      GIM_Reject,
    // Label 430: @11355
    GIM_Reject,
    // Label 385: @11356
    GIM_Try, /*On fail goto*//*Label 434*/ 11464,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 435*/ 11389, // Rule ID 3645 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KXORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3645,
        GIR_Done,
      // Label 435: @11389
      GIM_Try, /*On fail goto*//*Label 436*/ 11463, // Rule ID 13918 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/11,
        // GIR_Coverage, 13918,
        GIR_Done,
      // Label 436: @11463
      GIM_Reject,
    // Label 434: @11464
    GIM_Reject,
    // Label 386: @11465
    GIM_Try, /*On fail goto*//*Label 437*/ 11496, // Rule ID 4996 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPXORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4996,
      GIR_Done,
    // Label 437: @11496
    GIM_Reject,
    // Label 387: @11497
    GIM_Try, /*On fail goto*//*Label 438*/ 11528, // Rule ID 3646 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (xor:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KXORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3646,
      GIR_Done,
    // Label 438: @11528
    GIM_Reject,
    // Label 388: @11529
    GIM_Try, /*On fail goto*//*Label 439*/ 11560, // Rule ID 3647 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (xor:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KXORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3647,
      GIR_Done,
    // Label 439: @11560
    GIM_Reject,
    // Label 389: @11561
    GIM_Try, /*On fail goto*//*Label 440*/ 11592, // Rule ID 3648 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (xor:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KXORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3648,
      GIR_Done,
    // Label 440: @11592
    GIM_Reject,
    // Label 390: @11593
    GIM_Reject,
    // Label 6: @11594
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 25, /*)*//*default:*//*Label 458*/ 14942,
    /*GILLT_s32*//*Label 441*/ 11622,
    /*GILLT_s64*//*Label 442*/ 11784, 0,
    /*GILLT_s128*//*Label 443*/ 11946, 0,
    /*GILLT_v2s64*//*Label 444*/ 12002, 0,
    /*GILLT_v4s32*//*Label 445*/ 12323,
    /*GILLT_v4s64*//*Label 446*/ 12644, 0,
    /*GILLT_v8s16*//*Label 447*/ 12965,
    /*GILLT_v8s32*//*Label 448*/ 13126,
    /*GILLT_v8s64*//*Label 449*/ 13447, 0,
    /*GILLT_v16s8*//*Label 450*/ 13768,
    /*GILLT_v16s16*//*Label 451*/ 13929,
    /*GILLT_v16s32*//*Label 452*/ 14090,
    /*GILLT_v32s1*//*Label 453*/ 14411,
    /*GILLT_v32s8*//*Label 454*/ 14435,
    /*GILLT_v32s16*//*Label 455*/ 14596,
    /*GILLT_v64s1*//*Label 456*/ 14757,
    /*GILLT_v64s8*//*Label 457*/ 14781,
    // Label 441: @11622
    GIM_Try, /*On fail goto*//*Label 459*/ 11645, // Rule ID 2226 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2226,
      GIR_Done,
    // Label 459: @11645
    GIM_Try, /*On fail goto*//*Label 460*/ 11668, // Rule ID 2228 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (MOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2228,
      GIR_Done,
    // Label 460: @11668
    GIM_Try, /*On fail goto*//*Label 461*/ 11691, // Rule ID 2242 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VMOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2242,
      GIR_Done,
    // Label 461: @11691
    GIM_Try, /*On fail goto*//*Label 462*/ 11714, // Rule ID 2244 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (MOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2244,
      GIR_Done,
    // Label 462: @11714
    GIM_Try, /*On fail goto*//*Label 463*/ 11737, // Rule ID 3818 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSZrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3818,
      GIR_Done,
    // Label 463: @11737
    GIM_Try, /*On fail goto*//*Label 464*/ 11760, // Rule ID 3824 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (bitconvert:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VMOVSS2DIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3824,
      GIR_Done,
    // Label 464: @11760
    GIM_Try, /*On fail goto*//*Label 465*/ 11783, // Rule ID 13876 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (bitconvert:{ *:[i32] } VK32:{ *:[v32i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK32:{ *:[v32i1] }:$src, GR32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/26,
      // GIR_Coverage, 13876,
      GIR_Done,
    // Label 465: @11783
    GIM_Reject,
    // Label 442: @11784
    GIM_Try, /*On fail goto*//*Label 466*/ 11807, // Rule ID 2221 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2221,
      GIR_Done,
    // Label 466: @11807
    GIM_Try, /*On fail goto*//*Label 467*/ 11830, // Rule ID 2225 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (MOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2225,
      GIR_Done,
    // Label 467: @11830
    GIM_Try, /*On fail goto*//*Label 468*/ 11853, // Rule ID 2237 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VMOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2237,
      GIR_Done,
    // Label 468: @11853
    GIM_Try, /*On fail goto*//*Label 469*/ 11876, // Rule ID 2240 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (MOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2240,
      GIR_Done,
    // Label 469: @11876
    GIM_Try, /*On fail goto*//*Label 470*/ 11899, // Rule ID 3814 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDZrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3814,
      GIR_Done,
    // Label 470: @11899
    GIM_Try, /*On fail goto*//*Label 471*/ 11922, // Rule ID 3816 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (bitconvert:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VMOVSDto64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3816,
      GIR_Done,
    // Label 471: @11922
    GIM_Try, /*On fail goto*//*Label 472*/ 11945, // Rule ID 13878 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (bitconvert:{ *:[i64] } VK64:{ *:[v64i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i64] } VK64:{ *:[v64i1] }:$src, GR64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR64*/44,
      // GIR_Coverage, 13878,
      GIR_Done,
    // Label 472: @11945
    GIM_Reject,
    // Label 443: @11946
    GIM_Try, /*On fail goto*//*Label 473*/ 12001,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s128,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR128RegClassID,
      GIM_Try, /*On fail goto*//*Label 474*/ 11980, // Rule ID 15926 //
        // (bitconvert:{ *:[f128] } FR128:{ *:[i128] }:$src)  =>  FR128:{ *:[f128] }:$src
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC FR128*/73,
        // GIR_Coverage, 15926,
        GIR_Done,
      // Label 474: @11980
      GIM_Try, /*On fail goto*//*Label 475*/ 12000, // Rule ID 15927 //
        // (bitconvert:{ *:[i128] } FR128:{ *:[f128] }:$src)  =>  FR128:{ *:[i128] }:$src
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC FR128*/73,
        // GIR_Coverage, 15927,
        GIR_Done,
      // Label 475: @12000
      GIM_Reject,
    // Label 473: @12001
    GIM_Reject,
    // Label 444: @12002
    GIM_Try, /*On fail goto*//*Label 476*/ 12034, // Rule ID 15896 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)  =>  VR128:{ *:[v2i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15896,
      GIR_Done,
    // Label 476: @12034
    GIM_Try, /*On fail goto*//*Label 477*/ 12066, // Rule ID 15897 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)  =>  VR128:{ *:[v2i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15897,
      GIR_Done,
    // Label 477: @12066
    GIM_Try, /*On fail goto*//*Label 478*/ 12098, // Rule ID 15898 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)  =>  VR128:{ *:[v2i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15898,
      GIR_Done,
    // Label 478: @12098
    GIM_Try, /*On fail goto*//*Label 479*/ 12130, // Rule ID 15899 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2i64] } VR128:{ *:[v2f64] }:$src)  =>  VR128:{ *:[v2i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15899,
      GIR_Done,
    // Label 479: @12130
    GIM_Try, /*On fail goto*//*Label 480*/ 12162, // Rule ID 15900 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2i64] } VR128:{ *:[v4f32] }:$src)  =>  VR128:{ *:[v2i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15900,
      GIR_Done,
    // Label 480: @12162
    GIM_Try, /*On fail goto*//*Label 481*/ 12194, // Rule ID 15921 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2f64] } VR128:{ *:[v2i64] }:$src)  =>  VR128:{ *:[v2f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15921,
      GIR_Done,
    // Label 481: @12194
    GIM_Try, /*On fail goto*//*Label 482*/ 12226, // Rule ID 15922 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2f64] } VR128:{ *:[v4i32] }:$src)  =>  VR128:{ *:[v2f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15922,
      GIR_Done,
    // Label 482: @12226
    GIM_Try, /*On fail goto*//*Label 483*/ 12258, // Rule ID 15923 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2f64] } VR128:{ *:[v8i16] }:$src)  =>  VR128:{ *:[v2f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15923,
      GIR_Done,
    // Label 483: @12258
    GIM_Try, /*On fail goto*//*Label 484*/ 12290, // Rule ID 15924 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2f64] } VR128:{ *:[v16i8] }:$src)  =>  VR128:{ *:[v2f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15924,
      GIR_Done,
    // Label 484: @12290
    GIM_Try, /*On fail goto*//*Label 485*/ 12322, // Rule ID 15925 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v2f64] } VR128:{ *:[v4f32] }:$src)  =>  VR128:{ *:[v2f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15925,
      GIR_Done,
    // Label 485: @12322
    GIM_Reject,
    // Label 445: @12323
    GIM_Try, /*On fail goto*//*Label 486*/ 12355, // Rule ID 15901 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4i32] } VR128:{ *:[v2i64] }:$src)  =>  VR128:{ *:[v4i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15901,
      GIR_Done,
    // Label 486: @12355
    GIM_Try, /*On fail goto*//*Label 487*/ 12387, // Rule ID 15902 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)  =>  VR128:{ *:[v4i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15902,
      GIR_Done,
    // Label 487: @12387
    GIM_Try, /*On fail goto*//*Label 488*/ 12419, // Rule ID 15903 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)  =>  VR128:{ *:[v4i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15903,
      GIR_Done,
    // Label 488: @12419
    GIM_Try, /*On fail goto*//*Label 489*/ 12451, // Rule ID 15904 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4i32] } VR128:{ *:[v2f64] }:$src)  =>  VR128:{ *:[v4i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15904,
      GIR_Done,
    // Label 489: @12451
    GIM_Try, /*On fail goto*//*Label 490*/ 12483, // Rule ID 15905 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  VR128:{ *:[v4i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15905,
      GIR_Done,
    // Label 490: @12483
    GIM_Try, /*On fail goto*//*Label 491*/ 12515, // Rule ID 15916 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4f32] } VR128:{ *:[v2i64] }:$src)  =>  VR128:{ *:[v4f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15916,
      GIR_Done,
    // Label 491: @12515
    GIM_Try, /*On fail goto*//*Label 492*/ 12547, // Rule ID 15917 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  VR128:{ *:[v4f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15917,
      GIR_Done,
    // Label 492: @12547
    GIM_Try, /*On fail goto*//*Label 493*/ 12579, // Rule ID 15918 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4f32] } VR128:{ *:[v8i16] }:$src)  =>  VR128:{ *:[v4f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15918,
      GIR_Done,
    // Label 493: @12579
    GIM_Try, /*On fail goto*//*Label 494*/ 12611, // Rule ID 15919 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4f32] } VR128:{ *:[v16i8] }:$src)  =>  VR128:{ *:[v4f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15919,
      GIR_Done,
    // Label 494: @12611
    GIM_Try, /*On fail goto*//*Label 495*/ 12643, // Rule ID 15920 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v4f32] } VR128:{ *:[v2f64] }:$src)  =>  VR128:{ *:[v4f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15920,
      GIR_Done,
    // Label 495: @12643
    GIM_Reject,
    // Label 446: @12644
    GIM_Try, /*On fail goto*//*Label 496*/ 12676, // Rule ID 15928 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4i64] } VR256:{ *:[v8i32] }:$src)  =>  VR256:{ *:[v4i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15928,
      GIR_Done,
    // Label 496: @12676
    GIM_Try, /*On fail goto*//*Label 497*/ 12708, // Rule ID 15929 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4i64] } VR256:{ *:[v16i16] }:$src)  =>  VR256:{ *:[v4i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15929,
      GIR_Done,
    // Label 497: @12708
    GIM_Try, /*On fail goto*//*Label 498*/ 12740, // Rule ID 15930 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4i64] } VR256:{ *:[v32i8] }:$src)  =>  VR256:{ *:[v4i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15930,
      GIR_Done,
    // Label 498: @12740
    GIM_Try, /*On fail goto*//*Label 499*/ 12772, // Rule ID 15931 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4i64] } VR256:{ *:[v8f32] }:$src)  =>  VR256:{ *:[v4i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15931,
      GIR_Done,
    // Label 499: @12772
    GIM_Try, /*On fail goto*//*Label 500*/ 12804, // Rule ID 15932 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4i64] } VR256:{ *:[v4f64] }:$src)  =>  VR256:{ *:[v4i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15932,
      GIR_Done,
    // Label 500: @12804
    GIM_Try, /*On fail goto*//*Label 501*/ 12836, // Rule ID 15953 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4f64] } VR256:{ *:[v4i64] }:$src)  =>  VR256:{ *:[v4f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15953,
      GIR_Done,
    // Label 501: @12836
    GIM_Try, /*On fail goto*//*Label 502*/ 12868, // Rule ID 15954 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4f64] } VR256:{ *:[v8i32] }:$src)  =>  VR256:{ *:[v4f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15954,
      GIR_Done,
    // Label 502: @12868
    GIM_Try, /*On fail goto*//*Label 503*/ 12900, // Rule ID 15955 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4f64] } VR256:{ *:[v16i16] }:$src)  =>  VR256:{ *:[v4f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15955,
      GIR_Done,
    // Label 503: @12900
    GIM_Try, /*On fail goto*//*Label 504*/ 12932, // Rule ID 15956 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4f64] } VR256:{ *:[v32i8] }:$src)  =>  VR256:{ *:[v4f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15956,
      GIR_Done,
    // Label 504: @12932
    GIM_Try, /*On fail goto*//*Label 505*/ 12964, // Rule ID 15957 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v4f64] } VR256:{ *:[v8f32] }:$src)  =>  VR256:{ *:[v4f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15957,
      GIR_Done,
    // Label 505: @12964
    GIM_Reject,
    // Label 447: @12965
    GIM_Try, /*On fail goto*//*Label 506*/ 12997, // Rule ID 15906 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v8i16] } VR128:{ *:[v2i64] }:$src)  =>  VR128:{ *:[v8i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15906,
      GIR_Done,
    // Label 506: @12997
    GIM_Try, /*On fail goto*//*Label 507*/ 13029, // Rule ID 15907 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v8i16] } VR128:{ *:[v4i32] }:$src)  =>  VR128:{ *:[v8i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15907,
      GIR_Done,
    // Label 507: @13029
    GIM_Try, /*On fail goto*//*Label 508*/ 13061, // Rule ID 15908 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)  =>  VR128:{ *:[v8i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15908,
      GIR_Done,
    // Label 508: @13061
    GIM_Try, /*On fail goto*//*Label 509*/ 13093, // Rule ID 15909 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v8i16] } VR128:{ *:[v2f64] }:$src)  =>  VR128:{ *:[v8i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15909,
      GIR_Done,
    // Label 509: @13093
    GIM_Try, /*On fail goto*//*Label 510*/ 13125, // Rule ID 15910 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v8i16] } VR128:{ *:[v4f32] }:$src)  =>  VR128:{ *:[v8i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15910,
      GIR_Done,
    // Label 510: @13125
    GIM_Reject,
    // Label 448: @13126
    GIM_Try, /*On fail goto*//*Label 511*/ 13158, // Rule ID 15933 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8i32] } VR256:{ *:[v4i64] }:$src)  =>  VR256:{ *:[v8i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15933,
      GIR_Done,
    // Label 511: @13158
    GIM_Try, /*On fail goto*//*Label 512*/ 13190, // Rule ID 15934 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8i32] } VR256:{ *:[v16i16] }:$src)  =>  VR256:{ *:[v8i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15934,
      GIR_Done,
    // Label 512: @13190
    GIM_Try, /*On fail goto*//*Label 513*/ 13222, // Rule ID 15935 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8i32] } VR256:{ *:[v32i8] }:$src)  =>  VR256:{ *:[v8i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15935,
      GIR_Done,
    // Label 513: @13222
    GIM_Try, /*On fail goto*//*Label 514*/ 13254, // Rule ID 15936 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8i32] } VR256:{ *:[v4f64] }:$src)  =>  VR256:{ *:[v8i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15936,
      GIR_Done,
    // Label 514: @13254
    GIM_Try, /*On fail goto*//*Label 515*/ 13286, // Rule ID 15937 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)  =>  VR256:{ *:[v8i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15937,
      GIR_Done,
    // Label 515: @13286
    GIM_Try, /*On fail goto*//*Label 516*/ 13318, // Rule ID 15948 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8f32] } VR256:{ *:[v4i64] }:$src)  =>  VR256:{ *:[v8f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15948,
      GIR_Done,
    // Label 516: @13318
    GIM_Try, /*On fail goto*//*Label 517*/ 13350, // Rule ID 15949 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)  =>  VR256:{ *:[v8f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15949,
      GIR_Done,
    // Label 517: @13350
    GIM_Try, /*On fail goto*//*Label 518*/ 13382, // Rule ID 15950 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8f32] } VR256:{ *:[v16i16] }:$src)  =>  VR256:{ *:[v8f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15950,
      GIR_Done,
    // Label 518: @13382
    GIM_Try, /*On fail goto*//*Label 519*/ 13414, // Rule ID 15951 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8f32] } VR256:{ *:[v32i8] }:$src)  =>  VR256:{ *:[v8f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15951,
      GIR_Done,
    // Label 519: @13414
    GIM_Try, /*On fail goto*//*Label 520*/ 13446, // Rule ID 15952 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v8f32] } VR256:{ *:[v4f64] }:$src)  =>  VR256:{ *:[v8f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15952,
      GIR_Done,
    // Label 520: @13446
    GIM_Reject,
    // Label 449: @13447
    GIM_Try, /*On fail goto*//*Label 521*/ 13479, // Rule ID 15958 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  VR512:{ *:[v8f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15958,
      GIR_Done,
    // Label 521: @13479
    GIM_Try, /*On fail goto*//*Label 522*/ 13511, // Rule ID 15959 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8f64] } VR512:{ *:[v16i32] }:$src)  =>  VR512:{ *:[v8f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15959,
      GIR_Done,
    // Label 522: @13511
    GIM_Try, /*On fail goto*//*Label 523*/ 13543, // Rule ID 15960 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8f64] } VR512:{ *:[v32i16] }:$src)  =>  VR512:{ *:[v8f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15960,
      GIR_Done,
    // Label 523: @13543
    GIM_Try, /*On fail goto*//*Label 524*/ 13575, // Rule ID 15961 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8f64] } VR512:{ *:[v64i8] }:$src)  =>  VR512:{ *:[v8f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15961,
      GIR_Done,
    // Label 524: @13575
    GIM_Try, /*On fail goto*//*Label 525*/ 13607, // Rule ID 15962 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8f64] } VR512:{ *:[v16f32] }:$src)  =>  VR512:{ *:[v8f64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15962,
      GIR_Done,
    // Label 525: @13607
    GIM_Try, /*On fail goto*//*Label 526*/ 13639, // Rule ID 15968 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8i64] } VR512:{ *:[v16i32] }:$src)  =>  VR512:{ *:[v8i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15968,
      GIR_Done,
    // Label 526: @13639
    GIM_Try, /*On fail goto*//*Label 527*/ 13671, // Rule ID 15969 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8i64] } VR512:{ *:[v32i16] }:$src)  =>  VR512:{ *:[v8i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15969,
      GIR_Done,
    // Label 527: @13671
    GIM_Try, /*On fail goto*//*Label 528*/ 13703, // Rule ID 15970 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8i64] } VR512:{ *:[v64i8] }:$src)  =>  VR512:{ *:[v8i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15970,
      GIR_Done,
    // Label 528: @13703
    GIM_Try, /*On fail goto*//*Label 529*/ 13735, // Rule ID 15971 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  VR512:{ *:[v8i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15971,
      GIR_Done,
    // Label 529: @13735
    GIM_Try, /*On fail goto*//*Label 530*/ 13767, // Rule ID 15972 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v8i64] } VR512:{ *:[v16f32] }:$src)  =>  VR512:{ *:[v8i64] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15972,
      GIR_Done,
    // Label 530: @13767
    GIM_Reject,
    // Label 450: @13768
    GIM_Try, /*On fail goto*//*Label 531*/ 13800, // Rule ID 15911 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v16i8] } VR128:{ *:[v2i64] }:$src)  =>  VR128:{ *:[v16i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15911,
      GIR_Done,
    // Label 531: @13800
    GIM_Try, /*On fail goto*//*Label 532*/ 13832, // Rule ID 15912 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v16i8] } VR128:{ *:[v4i32] }:$src)  =>  VR128:{ *:[v16i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15912,
      GIR_Done,
    // Label 532: @13832
    GIM_Try, /*On fail goto*//*Label 533*/ 13864, // Rule ID 15913 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v16i8] } VR128:{ *:[v8i16] }:$src)  =>  VR128:{ *:[v16i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15913,
      GIR_Done,
    // Label 533: @13864
    GIM_Try, /*On fail goto*//*Label 534*/ 13896, // Rule ID 15914 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v16i8] } VR128:{ *:[v2f64] }:$src)  =>  VR128:{ *:[v16i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15914,
      GIR_Done,
    // Label 534: @13896
    GIM_Try, /*On fail goto*//*Label 535*/ 13928, // Rule ID 15915 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (bitconvert:{ *:[v16i8] } VR128:{ *:[v4f32] }:$src)  =>  VR128:{ *:[v16i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR128*/74,
      // GIR_Coverage, 15915,
      GIR_Done,
    // Label 535: @13928
    GIM_Reject,
    // Label 451: @13929
    GIM_Try, /*On fail goto*//*Label 536*/ 13961, // Rule ID 15938 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v16i16] } VR256:{ *:[v4i64] }:$src)  =>  VR256:{ *:[v16i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15938,
      GIR_Done,
    // Label 536: @13961
    GIM_Try, /*On fail goto*//*Label 537*/ 13993, // Rule ID 15939 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v16i16] } VR256:{ *:[v8i32] }:$src)  =>  VR256:{ *:[v16i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15939,
      GIR_Done,
    // Label 537: @13993
    GIM_Try, /*On fail goto*//*Label 538*/ 14025, // Rule ID 15940 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v16i16] } VR256:{ *:[v32i8] }:$src)  =>  VR256:{ *:[v16i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15940,
      GIR_Done,
    // Label 538: @14025
    GIM_Try, /*On fail goto*//*Label 539*/ 14057, // Rule ID 15941 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v16i16] } VR256:{ *:[v4f64] }:$src)  =>  VR256:{ *:[v16i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15941,
      GIR_Done,
    // Label 539: @14057
    GIM_Try, /*On fail goto*//*Label 540*/ 14089, // Rule ID 15942 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v16i16] } VR256:{ *:[v8f32] }:$src)  =>  VR256:{ *:[v16i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15942,
      GIR_Done,
    // Label 540: @14089
    GIM_Reject,
    // Label 452: @14090
    GIM_Try, /*On fail goto*//*Label 541*/ 14122, // Rule ID 15963 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16f32] } VR512:{ *:[v8i64] }:$src)  =>  VR512:{ *:[v16f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15963,
      GIR_Done,
    // Label 541: @14122
    GIM_Try, /*On fail goto*//*Label 542*/ 14154, // Rule ID 15964 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  VR512:{ *:[v16f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15964,
      GIR_Done,
    // Label 542: @14154
    GIM_Try, /*On fail goto*//*Label 543*/ 14186, // Rule ID 15965 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16f32] } VR512:{ *:[v32i16] }:$src)  =>  VR512:{ *:[v16f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15965,
      GIR_Done,
    // Label 543: @14186
    GIM_Try, /*On fail goto*//*Label 544*/ 14218, // Rule ID 15966 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16f32] } VR512:{ *:[v64i8] }:$src)  =>  VR512:{ *:[v16f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15966,
      GIR_Done,
    // Label 544: @14218
    GIM_Try, /*On fail goto*//*Label 545*/ 14250, // Rule ID 15967 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16f32] } VR512:{ *:[v8f64] }:$src)  =>  VR512:{ *:[v16f32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15967,
      GIR_Done,
    // Label 545: @14250
    GIM_Try, /*On fail goto*//*Label 546*/ 14282, // Rule ID 15973 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16i32] } VR512:{ *:[v8i64] }:$src)  =>  VR512:{ *:[v16i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15973,
      GIR_Done,
    // Label 546: @14282
    GIM_Try, /*On fail goto*//*Label 547*/ 14314, // Rule ID 15974 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  VR512:{ *:[v16i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15974,
      GIR_Done,
    // Label 547: @14314
    GIM_Try, /*On fail goto*//*Label 548*/ 14346, // Rule ID 15975 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16i32] } VR512:{ *:[v32i16] }:$src)  =>  VR512:{ *:[v16i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15975,
      GIR_Done,
    // Label 548: @14346
    GIM_Try, /*On fail goto*//*Label 549*/ 14378, // Rule ID 15976 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16i32] } VR512:{ *:[v64i8] }:$src)  =>  VR512:{ *:[v16i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15976,
      GIR_Done,
    // Label 549: @14378
    GIM_Try, /*On fail goto*//*Label 550*/ 14410, // Rule ID 15977 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v16i32] } VR512:{ *:[v8f64] }:$src)  =>  VR512:{ *:[v16i32] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15977,
      GIR_Done,
    // Label 550: @14410
    GIM_Reject,
    // Label 453: @14411
    GIM_Try, /*On fail goto*//*Label 551*/ 14434, // Rule ID 13875 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[v32i1] } GR32:{ *:[i32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v32i1] } GR32:{ *:[i32] }:$src, VK32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK32*/30,
      // GIR_Coverage, 13875,
      GIR_Done,
    // Label 551: @14434
    GIM_Reject,
    // Label 454: @14435
    GIM_Try, /*On fail goto*//*Label 552*/ 14467, // Rule ID 15943 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v32i8] } VR256:{ *:[v4i64] }:$src)  =>  VR256:{ *:[v32i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15943,
      GIR_Done,
    // Label 552: @14467
    GIM_Try, /*On fail goto*//*Label 553*/ 14499, // Rule ID 15944 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v32i8] } VR256:{ *:[v8i32] }:$src)  =>  VR256:{ *:[v32i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15944,
      GIR_Done,
    // Label 553: @14499
    GIM_Try, /*On fail goto*//*Label 554*/ 14531, // Rule ID 15945 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v32i8] } VR256:{ *:[v16i16] }:$src)  =>  VR256:{ *:[v32i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15945,
      GIR_Done,
    // Label 554: @14531
    GIM_Try, /*On fail goto*//*Label 555*/ 14563, // Rule ID 15946 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v32i8] } VR256:{ *:[v4f64] }:$src)  =>  VR256:{ *:[v32i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15946,
      GIR_Done,
    // Label 555: @14563
    GIM_Try, /*On fail goto*//*Label 556*/ 14595, // Rule ID 15947 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (bitconvert:{ *:[v32i8] } VR256:{ *:[v8f32] }:$src)  =>  VR256:{ *:[v32i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR256*/79,
      // GIR_Coverage, 15947,
      GIR_Done,
    // Label 556: @14595
    GIM_Reject,
    // Label 455: @14596
    GIM_Try, /*On fail goto*//*Label 557*/ 14628, // Rule ID 15978 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v32i16] } VR512:{ *:[v8i64] }:$src)  =>  VR512:{ *:[v32i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15978,
      GIR_Done,
    // Label 557: @14628
    GIM_Try, /*On fail goto*//*Label 558*/ 14660, // Rule ID 15979 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v32i16] } VR512:{ *:[v16i32] }:$src)  =>  VR512:{ *:[v32i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15979,
      GIR_Done,
    // Label 558: @14660
    GIM_Try, /*On fail goto*//*Label 559*/ 14692, // Rule ID 15980 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v32i16] } VR512:{ *:[v64i8] }:$src)  =>  VR512:{ *:[v32i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15980,
      GIR_Done,
    // Label 559: @14692
    GIM_Try, /*On fail goto*//*Label 560*/ 14724, // Rule ID 15981 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v32i16] } VR512:{ *:[v8f64] }:$src)  =>  VR512:{ *:[v32i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15981,
      GIR_Done,
    // Label 560: @14724
    GIM_Try, /*On fail goto*//*Label 561*/ 14756, // Rule ID 15982 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v32i16] } VR512:{ *:[v16f32] }:$src)  =>  VR512:{ *:[v32i16] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15982,
      GIR_Done,
    // Label 561: @14756
    GIM_Reject,
    // Label 456: @14757
    GIM_Try, /*On fail goto*//*Label 562*/ 14780, // Rule ID 13877 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[v64i1] } GR64:{ *:[i64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v64i1] } GR64:{ *:[i64] }:$src, VK64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK64*/54,
      // GIR_Coverage, 13877,
      GIR_Done,
    // Label 562: @14780
    GIM_Reject,
    // Label 457: @14781
    GIM_Try, /*On fail goto*//*Label 563*/ 14813, // Rule ID 15983 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v64i8] } VR512:{ *:[v8i64] }:$src)  =>  VR512:{ *:[v64i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15983,
      GIR_Done,
    // Label 563: @14813
    GIM_Try, /*On fail goto*//*Label 564*/ 14845, // Rule ID 15984 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v64i8] } VR512:{ *:[v16i32] }:$src)  =>  VR512:{ *:[v64i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15984,
      GIR_Done,
    // Label 564: @14845
    GIM_Try, /*On fail goto*//*Label 565*/ 14877, // Rule ID 15985 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v64i8] } VR512:{ *:[v32i16] }:$src)  =>  VR512:{ *:[v64i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15985,
      GIR_Done,
    // Label 565: @14877
    GIM_Try, /*On fail goto*//*Label 566*/ 14909, // Rule ID 15986 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v64i8] } VR512:{ *:[v8f64] }:$src)  =>  VR512:{ *:[v64i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15986,
      GIR_Done,
    // Label 566: @14909
    GIM_Try, /*On fail goto*//*Label 567*/ 14941, // Rule ID 15987 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (bitconvert:{ *:[v64i8] } VR512:{ *:[v16f32] }:$src)  =>  VR512:{ *:[v64i8] }:$src
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VR512*/82,
      // GIR_Coverage, 15987,
      GIR_Done,
    // Label 567: @14941
    GIM_Reject,
    // Label 458: @14942
    GIM_Reject,
    // Label 7: @14943
    GIM_Try, /*On fail goto*//*Label 568*/ 17269,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 569*/ 14988, // Rule ID 1157 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6777:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHSUBWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1157,
        GIR_Done,
      // Label 569: @14988
      GIM_Try, /*On fail goto*//*Label 570*/ 15028, // Rule ID 1159 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6776:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHSUBDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1159,
        GIR_Done,
      // Label 570: @15028
      GIM_Try, /*On fail goto*//*Label 571*/ 15068, // Rule ID 1161 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6775:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHSUBBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1161,
        GIR_Done,
      // Label 571: @15068
      GIM_Try, /*On fail goto*//*Label 572*/ 15108, // Rule ID 1163 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6774:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1163,
        GIR_Done,
      // Label 572: @15108
      GIM_Try, /*On fail goto*//*Label 573*/ 15148, // Rule ID 1165 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6773:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1165,
        GIR_Done,
      // Label 573: @15148
      GIM_Try, /*On fail goto*//*Label 574*/ 15188, // Rule ID 1167 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6772:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1167,
        GIR_Done,
      // Label 574: @15188
      GIM_Try, /*On fail goto*//*Label 575*/ 15228, // Rule ID 1169 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6771:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1169,
        GIR_Done,
      // Label 575: @15228
      GIM_Try, /*On fail goto*//*Label 576*/ 15268, // Rule ID 1171 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddudq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6770:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDUDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1171,
        GIR_Done,
      // Label 576: @15268
      GIM_Try, /*On fail goto*//*Label 577*/ 15308, // Rule ID 1173 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6769:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1173,
        GIR_Done,
      // Label 577: @15308
      GIM_Try, /*On fail goto*//*Label 578*/ 15348, // Rule ID 1175 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6768:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1175,
        GIR_Done,
      // Label 578: @15348
      GIM_Try, /*On fail goto*//*Label 579*/ 15388, // Rule ID 1177 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6767:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1177,
        GIR_Done,
      // Label 579: @15388
      GIM_Try, /*On fail goto*//*Label 580*/ 15428, // Rule ID 1179 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadddq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6766:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1179,
        GIR_Done,
      // Label 580: @15428
      GIM_Try, /*On fail goto*//*Label 581*/ 15468, // Rule ID 1181 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6765:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1181,
        GIR_Done,
      // Label 581: @15468
      GIM_Try, /*On fail goto*//*Label 582*/ 15508, // Rule ID 1183 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6764:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1183,
        GIR_Done,
      // Label 582: @15508
      GIM_Try, /*On fail goto*//*Label 583*/ 15548, // Rule ID 1185 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6763:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1185,
        GIR_Done,
      // Label 583: @15548
      GIM_Try, /*On fail goto*//*Label 584*/ 15588, // Rule ID 1187 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6750:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZSSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1187,
        GIR_Done,
      // Label 584: @15588
      GIM_Try, /*On fail goto*//*Label 585*/ 15628, // Rule ID 1189 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6747:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1189,
        GIR_Done,
      // Label 585: @15628
      GIM_Try, /*On fail goto*//*Label 586*/ 15668, // Rule ID 1191 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 6748:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src)  =>  (VFRCZPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1191,
        GIR_Done,
      // Label 586: @15668
      GIM_Try, /*On fail goto*//*Label 587*/ 15708, // Rule ID 1193 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6749:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZSDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1193,
        GIR_Done,
      // Label 587: @15708
      GIM_Try, /*On fail goto*//*Label 588*/ 15748, // Rule ID 1195 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6745:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1195,
        GIR_Done,
      // Label 588: @15748
      GIM_Try, /*On fail goto*//*Label 589*/ 15788, // Rule ID 1197 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 6746:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src)  =>  (VFRCZPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1197,
        GIR_Done,
      // Label 589: @15788
      GIM_Try, /*On fail goto*//*Label 590*/ 15828, // Rule ID 1391 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6580:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VCVTSD2SIrr_Int:{ *:[i32] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1391,
        GIR_Done,
      // Label 590: @15828
      GIM_Try, /*On fail goto*//*Label 591*/ 15868, // Rule ID 1393 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6581:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VCVTSD2SI64rr_Int:{ *:[i64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1393,
        GIR_Done,
      // Label 591: @15868
      GIM_Try, /*On fail goto*//*Label 592*/ 15908, // Rule ID 1395 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6580:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (CVTSD2SIrr_Int:{ *:[i32] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSD2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1395,
        GIR_Done,
      // Label 592: @15908
      GIM_Try, /*On fail goto*//*Label 593*/ 15948, // Rule ID 1397 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6581:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (CVTSD2SI64rr_Int:{ *:[i64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSD2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1397,
        GIR_Done,
      // Label 593: @15948
      GIM_Try, /*On fail goto*//*Label 594*/ 15988, // Rule ID 1399 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6545:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VCVTTSS2SIrr_Int:{ *:[i32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSS2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1399,
        GIR_Done,
      // Label 594: @15988
      GIM_Try, /*On fail goto*//*Label 595*/ 16028, // Rule ID 1401 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6546:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VCVTTSS2SI64rr_Int:{ *:[i64] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSS2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1401,
        GIR_Done,
      // Label 595: @16028
      GIM_Try, /*On fail goto*//*Label 596*/ 16068, // Rule ID 1403 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6585:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VCVTTSD2SIrr_Int:{ *:[i32] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSD2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1403,
        GIR_Done,
      // Label 596: @16068
      GIM_Try, /*On fail goto*//*Label 597*/ 16108, // Rule ID 1405 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6586:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VCVTTSD2SI64rr_Int:{ *:[i64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSD2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1405,
        GIR_Done,
      // Label 597: @16108
      GIM_Try, /*On fail goto*//*Label 598*/ 16148, // Rule ID 1407 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6545:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (CVTTSS2SIrr_Int:{ *:[i32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTTSS2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1407,
        GIR_Done,
      // Label 598: @16148
      GIM_Try, /*On fail goto*//*Label 599*/ 16188, // Rule ID 1409 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6546:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (CVTTSS2SI64rr_Int:{ *:[i64] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTTSS2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1409,
        GIR_Done,
      // Label 599: @16188
      GIM_Try, /*On fail goto*//*Label 600*/ 16228, // Rule ID 1411 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6585:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (CVTTSD2SIrr_Int:{ *:[i32] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTTSD2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1411,
        GIR_Done,
      // Label 600: @16228
      GIM_Try, /*On fail goto*//*Label 601*/ 16268, // Rule ID 1413 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6586:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (CVTTSD2SI64rr_Int:{ *:[i64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTTSD2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1413,
        GIR_Done,
      // Label 601: @16268
      GIM_Try, /*On fail goto*//*Label 602*/ 16308, // Rule ID 1415 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6541:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VCVTSS2SIrr_Int:{ *:[i32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1415,
        GIR_Done,
      // Label 602: @16308
      GIM_Try, /*On fail goto*//*Label 603*/ 16348, // Rule ID 1417 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6542:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VCVTSS2SI64rr_Int:{ *:[i64] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1417,
        GIR_Done,
      // Label 603: @16348
      GIM_Try, /*On fail goto*//*Label 604*/ 16388, // Rule ID 1419 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6541:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (CVTSS2SIrr_Int:{ *:[i32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSS2SIrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1419,
        GIR_Done,
      // Label 604: @16388
      GIM_Try, /*On fail goto*//*Label 605*/ 16428, // Rule ID 1421 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6542:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (CVTSS2SI64rr_Int:{ *:[i64] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSS2SI64rr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1421,
        GIR_Done,
      // Label 605: @16428
      GIM_Try, /*On fail goto*//*Label 606*/ 16468, // Rule ID 2655 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5495:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (VAESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2655,
        GIR_Done,
      // Label 606: @16468
      GIM_Try, /*On fail goto*//*Label 607*/ 16508, // Rule ID 2657 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5495:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (AESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2657,
        GIR_Done,
      // Label 607: @16508
      GIM_Try, /*On fail goto*//*Label 608*/ 16552, // Rule ID 12779 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_sqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6560:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (SQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12779,
        GIR_Done,
      // Label 608: @16552
      GIM_Try, /*On fail goto*//*Label 609*/ 16596, // Rule ID 12782 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_sqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6560:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12782,
        GIR_Done,
      // Label 609: @16596
      GIM_Try, /*On fail goto*//*Label 610*/ 16640, // Rule ID 12785 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_sqrt_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6629:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (SQRTSDr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src, VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SQRTSDr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12785,
        GIR_Done,
      // Label 610: @16640
      GIM_Try, /*On fail goto*//*Label 611*/ 16684, // Rule ID 12788 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_sqrt_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6629:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VSQRTSDr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src, VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VSQRTSDr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12788,
        GIR_Done,
      // Label 611: @16684
      GIM_Try, /*On fail goto*//*Label 612*/ 16728, // Rule ID 12791 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6557:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12791,
        GIR_Done,
      // Label 612: @16728
      GIM_Try, /*On fail goto*//*Label 613*/ 16772, // Rule ID 12794 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6557:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12794,
        GIR_Done,
      // Label 613: @16772
      GIM_Try, /*On fail goto*//*Label 614*/ 16816, // Rule ID 12797 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6555:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12797,
        GIR_Done,
      // Label 614: @16816
      GIM_Try, /*On fail goto*//*Label 615*/ 16860, // Rule ID 12800 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6555:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12800,
        GIR_Done,
      // Label 615: @16860
      GIM_Try, /*On fail goto*//*Label 616*/ 16900, // Rule ID 14845 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6541:{ *:[iPTR] }, VR128X:{ *:[v4f32] }:$src)  =>  (VCVTSS2SIZrr_Int:{ *:[i32] } VR128X:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SIZrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14845,
        GIR_Done,
      // Label 616: @16900
      GIM_Try, /*On fail goto*//*Label 617*/ 16940, // Rule ID 14847 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvtss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6542:{ *:[iPTR] }, VR128X:{ *:[v4f32] }:$src)  =>  (VCVTSS2SI64Zrr_Int:{ *:[i64] } VR128X:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SI64Zrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14847,
        GIR_Done,
      // Label 617: @16940
      GIM_Try, /*On fail goto*//*Label 618*/ 16980, // Rule ID 14849 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6580:{ *:[iPTR] }, VR128X:{ *:[v2f64] }:$src)  =>  (VCVTSD2SIZrr_Int:{ *:[i32] } VR128X:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SIZrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14849,
        GIR_Done,
      // Label 618: @16980
      GIM_Try, /*On fail goto*//*Label 619*/ 17020, // Rule ID 14851 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6581:{ *:[iPTR] }, VR128X:{ *:[v2f64] }:$src)  =>  (VCVTSD2SI64Zrr_Int:{ *:[i64] } VR128X:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SI64Zrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14851,
        GIR_Done,
      // Label 619: @17020
      GIM_Try, /*On fail goto*//*Label 620*/ 17060, // Rule ID 14869 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6545:{ *:[iPTR] }, VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTSS2SIZrr_Int:{ *:[i32] } VR128X:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSS2SIZrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14869,
        GIR_Done,
      // Label 620: @17060
      GIM_Try, /*On fail goto*//*Label 621*/ 17100, // Rule ID 14871 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cvttss2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6546:{ *:[iPTR] }, VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTSS2SI64Zrr_Int:{ *:[i64] } VR128X:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSS2SI64Zrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14871,
        GIR_Done,
      // Label 621: @17100
      GIM_Try, /*On fail goto*//*Label 622*/ 17140, // Rule ID 14873 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6585:{ *:[iPTR] }, VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTSD2SIZrr_Int:{ *:[i32] } VR128X:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSD2SIZrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14873,
        GIR_Done,
      // Label 622: @17140
      GIM_Try, /*On fail goto*//*Label 623*/ 17180, // Rule ID 14875 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvttsd2si64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6586:{ *:[iPTR] }, VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTSD2SI64Zrr_Int:{ *:[i64] } VR128X:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTTSD2SI64Zrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14875,
        GIR_Done,
      // Label 623: @17180
      GIM_Try, /*On fail goto*//*Label 624*/ 17224, // Rule ID 14933 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_sqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6560:{ *:[iPTR] }, VR128X:{ *:[v4f32] }:$src)  =>  (VSQRTSSZr_Int:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src, VR128X:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VSQRTSSZr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14933,
        GIR_Done,
      // Label 624: @17224
      GIM_Try, /*On fail goto*//*Label 625*/ 17268, // Rule ID 14937 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_sqrt_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6629:{ *:[iPTR] }, VR128X:{ *:[v2f64] }:$src)  =>  (VSQRTSDZr_Int:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src, VR128X:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VSQRTSDZr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14937,
        GIR_Done,
      // Label 625: @17268
      GIM_Reject,
    // Label 568: @17269
    GIM_Try, /*On fail goto*//*Label 626*/ 20299,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 627*/ 17330, // Rule ID 2659 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5496:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (VAESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2659,
        GIR_Done,
      // Label 627: @17330
      GIM_Try, /*On fail goto*//*Label 628*/ 17386, // Rule ID 2661 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5496:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2661,
        GIR_Done,
      // Label 628: @17386
      GIM_Try, /*On fail goto*//*Label 629*/ 17438, // Rule ID 82 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6378:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PDEP32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 82,
        GIR_Done,
      // Label 629: @17438
      GIM_Try, /*On fail goto*//*Label 630*/ 17490, // Rule ID 84 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6379:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PDEP64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 84,
        GIR_Done,
      // Label 630: @17490
      GIM_Try, /*On fail goto*//*Label 631*/ 17542, // Rule ID 86 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6380:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PEXT32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 86,
        GIR_Done,
      // Label 631: @17542
      GIM_Try, /*On fail goto*//*Label 632*/ 17594, // Rule ID 88 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6381:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PEXT64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 88,
        GIR_Done,
      // Label 632: @17594
      GIM_Try, /*On fail goto*//*Label 633*/ 17646, // Rule ID 1431 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6582:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VCVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1431,
        GIR_Done,
      // Label 633: @17646
      GIM_Try, /*On fail goto*//*Label 634*/ 17698, // Rule ID 1433 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6582:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (CVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1433,
        GIR_Done,
      // Label 634: @17698
      GIM_Try, /*On fail goto*//*Label 635*/ 17750, // Rule ID 2341 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6705:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2341,
        GIR_Done,
      // Label 635: @17750
      GIM_Try, /*On fail goto*//*Label 636*/ 17802, // Rule ID 2343 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6709:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2343,
        GIR_Done,
      // Label 636: @17802
      GIM_Try, /*On fail goto*//*Label 637*/ 17854, // Rule ID 2345 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6707:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2345,
        GIR_Done,
      // Label 637: @17854
      GIM_Try, /*On fail goto*//*Label 638*/ 17906, // Rule ID 2347 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6689:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2347,
        GIR_Done,
      // Label 638: @17906
      GIM_Try, /*On fail goto*//*Label 639*/ 17958, // Rule ID 2349 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6695:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2349,
        GIR_Done,
      // Label 639: @17958
      GIM_Try, /*On fail goto*//*Label 640*/ 18010, // Rule ID 2365 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_b,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5605:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSIGNBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2365,
        GIR_Done,
      // Label 640: @18010
      GIM_Try, /*On fail goto*//*Label 641*/ 18062, // Rule ID 2367 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_w,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5607:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSIGNWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2367,
        GIR_Done,
      // Label 641: @18062
      GIM_Try, /*On fail goto*//*Label 642*/ 18114, // Rule ID 2369 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_d,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i32] } 5606:{ *:[iPTR] }, VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSIGNDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2369,
        GIR_Done,
      // Label 642: @18114
      GIM_Try, /*On fail goto*//*Label 643*/ 18166, // Rule ID 2371 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phadd_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5592:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHADDSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2371,
        GIR_Done,
      // Label 643: @18166
      GIM_Try, /*On fail goto*//*Label 644*/ 18218, // Rule ID 2373 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phsub_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5595:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHSUBSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2373,
        GIR_Done,
      // Label 644: @18218
      GIM_Try, /*On fail goto*//*Label 645*/ 18270, // Rule ID 2383 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6705:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2383,
        GIR_Done,
      // Label 645: @18270
      GIM_Try, /*On fail goto*//*Label 646*/ 18322, // Rule ID 2385 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6709:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2385,
        GIR_Done,
      // Label 646: @18322
      GIM_Try, /*On fail goto*//*Label 647*/ 18374, // Rule ID 2387 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6707:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2387,
        GIR_Done,
      // Label 647: @18374
      GIM_Try, /*On fail goto*//*Label 648*/ 18426, // Rule ID 2391 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6689:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2391,
        GIR_Done,
      // Label 648: @18426
      GIM_Try, /*On fail goto*//*Label 649*/ 18478, // Rule ID 2393 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6695:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2393,
        GIR_Done,
      // Label 649: @18478
      GIM_Try, /*On fail goto*//*Label 650*/ 18530, // Rule ID 2610 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_8,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR8RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6663:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (CRC32r32r8:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2610,
        GIR_Done,
      // Label 650: @18530
      GIM_Try, /*On fail goto*//*Label 651*/ 18582, // Rule ID 2612 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_16,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR16RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6661:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (CRC32r32r16:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2612,
        GIR_Done,
      // Label 651: @18582
      GIM_Try, /*On fail goto*//*Label 652*/ 18634, // Rule ID 2614 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6662:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (CRC32r32r32:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2614,
        GIR_Done,
      // Label 652: @18634
      GIM_Try, /*On fail goto*//*Label 653*/ 18686, // Rule ID 2616 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_64_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6664:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (CRC32r64r64:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r64r64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2616,
        GIR_Done,
      // Label 653: @18686
      GIM_Try, /*On fail goto*//*Label 654*/ 18738, // Rule ID 2619 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1nexte,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6523:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1NEXTErr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1NEXTErr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2619,
        GIR_Done,
      // Label 654: @18738
      GIM_Try, /*On fail goto*//*Label 655*/ 18790, // Rule ID 2621 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6521:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2621,
        GIR_Done,
      // Label 655: @18790
      GIM_Try, /*On fail goto*//*Label 656*/ 18842, // Rule ID 2623 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6522:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2623,
        GIR_Done,
      // Label 656: @18842
      GIM_Try, /*On fail goto*//*Label 657*/ 18894, // Rule ID 2627 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6525:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2627,
        GIR_Done,
      // Label 657: @18894
      GIM_Try, /*On fail goto*//*Label 658*/ 18946, // Rule ID 2629 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6526:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2629,
        GIR_Done,
      // Label 658: @18946
      GIM_Try, /*On fail goto*//*Label 659*/ 18998, // Rule ID 2631 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5489:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2631,
        GIR_Done,
      // Label 659: @18998
      GIM_Try, /*On fail goto*//*Label 660*/ 19050, // Rule ID 2633 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5492:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2633,
        GIR_Done,
      // Label 660: @19050
      GIM_Try, /*On fail goto*//*Label 661*/ 19102, // Rule ID 2635 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5483:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2635,
        GIR_Done,
      // Label 661: @19102
      GIM_Try, /*On fail goto*//*Label 662*/ 19154, // Rule ID 2637 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5486:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2637,
        GIR_Done,
      // Label 662: @19154
      GIM_Try, /*On fail goto*//*Label 663*/ 19206, // Rule ID 2639 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5490:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2639,
        GIR_Done,
      // Label 663: @19206
      GIM_Try, /*On fail goto*//*Label 664*/ 19258, // Rule ID 2641 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5493:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2641,
        GIR_Done,
      // Label 664: @19258
      GIM_Try, /*On fail goto*//*Label 665*/ 19310, // Rule ID 2643 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5484:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2643,
        GIR_Done,
      // Label 665: @19310
      GIM_Try, /*On fail goto*//*Label 666*/ 19362, // Rule ID 2645 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5487:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2645,
        GIR_Done,
      // Label 666: @19362
      GIM_Try, /*On fail goto*//*Label 667*/ 19414, // Rule ID 2647 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5489:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2647,
        GIR_Done,
      // Label 667: @19414
      GIM_Try, /*On fail goto*//*Label 668*/ 19466, // Rule ID 2649 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5492:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2649,
        GIR_Done,
      // Label 668: @19466
      GIM_Try, /*On fail goto*//*Label 669*/ 19518, // Rule ID 2651 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5483:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2651,
        GIR_Done,
      // Label 669: @19518
      GIM_Try, /*On fail goto*//*Label 670*/ 19570, // Rule ID 2653 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5486:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2653,
        GIR_Done,
      // Label 670: @19570
      GIM_Try, /*On fail goto*//*Label 671*/ 19622, // Rule ID 2670 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_extrq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6679:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)  =>  (EXTRQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::EXTRQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2670,
        GIR_Done,
      // Label 671: @19622
      GIM_Try, /*On fail goto*//*Label 672*/ 19674, // Rule ID 2672 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_insertq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6681:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)  =>  (INSERTQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INSERTQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2672,
        GIR_Done,
      // Label 672: @19674
      GIM_Try, /*On fail goto*//*Label 673*/ 19726, // Rule ID 11356 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5489:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11356,
        GIR_Done,
      // Label 673: @19726
      GIM_Try, /*On fail goto*//*Label 674*/ 19778, // Rule ID 11358 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5490:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11358,
        GIR_Done,
      // Label 674: @19778
      GIM_Try, /*On fail goto*//*Label 675*/ 19830, // Rule ID 11360 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5491:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11360,
        GIR_Done,
      // Label 675: @19830
      GIM_Try, /*On fail goto*//*Label 676*/ 19882, // Rule ID 11362 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5492:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11362,
        GIR_Done,
      // Label 676: @19882
      GIM_Try, /*On fail goto*//*Label 677*/ 19934, // Rule ID 11364 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5493:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11364,
        GIR_Done,
      // Label 677: @19934
      GIM_Try, /*On fail goto*//*Label 678*/ 19986, // Rule ID 11366 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5494:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11366,
        GIR_Done,
      // Label 678: @19986
      GIM_Try, /*On fail goto*//*Label 679*/ 20038, // Rule ID 11368 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5483:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11368,
        GIR_Done,
      // Label 679: @20038
      GIM_Try, /*On fail goto*//*Label 680*/ 20090, // Rule ID 11370 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5484:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11370,
        GIR_Done,
      // Label 680: @20090
      GIM_Try, /*On fail goto*//*Label 681*/ 20142, // Rule ID 11372 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5485:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11372,
        GIR_Done,
      // Label 681: @20142
      GIM_Try, /*On fail goto*//*Label 682*/ 20194, // Rule ID 11374 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5486:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11374,
        GIR_Done,
      // Label 682: @20194
      GIM_Try, /*On fail goto*//*Label 683*/ 20246, // Rule ID 11376 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5487:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11376,
        GIR_Done,
      // Label 683: @20246
      GIM_Try, /*On fail goto*//*Label 684*/ 20298, // Rule ID 11378 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5488:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11378,
        GIR_Done,
      // Label 684: @20298
      GIM_Reject,
    // Label 626: @20299
    GIM_Try, /*On fail goto*//*Label 685*/ 22749,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/5,
      GIM_Try, /*On fail goto*//*Label 686*/ 20372, // Rule ID 1487 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6530:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1487,
        GIR_Done,
      // Label 686: @20372
      GIM_Try, /*On fail goto*//*Label 687*/ 20440, // Rule ID 1489 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6570:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1489,
        GIR_Done,
      // Label 687: @20440
      GIM_Try, /*On fail goto*//*Label 688*/ 20508, // Rule ID 1491 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6530:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1491,
        GIR_Done,
      // Label 688: @20508
      GIM_Try, /*On fail goto*//*Label 689*/ 20576, // Rule ID 1493 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6570:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1493,
        GIR_Done,
      // Label 689: @20576
      GIM_Try, /*On fail goto*//*Label 690*/ 20644, // Rule ID 2551 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6650:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2551,
        GIR_Done,
      // Label 690: @20644
      GIM_Try, /*On fail goto*//*Label 691*/ 20712, // Rule ID 2553 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6648:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2553,
        GIR_Done,
      // Label 691: @20712
      GIM_Try, /*On fail goto*//*Label 692*/ 20780, // Rule ID 2555 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6647:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2555,
        GIR_Done,
      // Label 692: @20780
      GIM_Try, /*On fail goto*//*Label 693*/ 20848, // Rule ID 2557 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_dp_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5508:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSYrri:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2557,
        GIR_Done,
      // Label 693: @20848
      GIM_Try, /*On fail goto*//*Label 694*/ 20916, // Rule ID 2559 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5579:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWYrri:{ *:[v16i16] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2559,
        GIR_Done,
      // Label 694: @20916
      GIM_Try, /*On fail goto*//*Label 695*/ 20984, // Rule ID 2561 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6650:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (MPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2561,
        GIR_Done,
      // Label 695: @20984
      GIM_Try, /*On fail goto*//*Label 696*/ 21052, // Rule ID 2563 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6648:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2563,
        GIR_Done,
      // Label 696: @21052
      GIM_Try, /*On fail goto*//*Label 697*/ 21120, // Rule ID 2565 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6647:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2565,
        GIR_Done,
      // Label 697: @21120
      GIM_Try, /*On fail goto*//*Label 698*/ 21188, // Rule ID 2617 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1rnds4,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6524:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (SHA1RNDS4rri:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1RNDS4rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2617,
        GIR_Done,
      // Label 698: @21188
      GIM_Try, /*On fail goto*//*Label 699*/ 21256, // Rule ID 2663 //
        GIM_CheckFeatures, GIFBS_HasPCLMUL_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6492:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (PCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2663,
        GIR_Done,
      // Label 699: @21256
      GIM_Try, /*On fail goto*//*Label 700*/ 21324, // Rule ID 2665 //
        GIM_CheckFeatures, GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6492:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2665,
        GIR_Done,
      // Label 700: @21324
      GIM_Try, /*On fail goto*//*Label 701*/ 21392, // Rule ID 2667 //
        GIM_CheckFeatures, GIFBS_HasVPCLMULQDQ_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6493:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2667,
        GIR_Done,
      // Label 701: @21392
      GIM_Try, /*On fail goto*//*Label 702*/ 21460, // Rule ID 11380 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i64] } 6494:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11380,
        GIR_Done,
      // Label 702: @21460
      GIM_Try, /*On fail goto*//*Label 703*/ 21528, // Rule ID 11382 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6492:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11382,
        GIR_Done,
      // Label 703: @21528
      GIM_Try, /*On fail goto*//*Label 704*/ 21596, // Rule ID 11384 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6493:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11384,
        GIR_Done,
      // Label 704: @21596
      GIM_Try, /*On fail goto*//*Label 705*/ 21660, // Rule ID 1243 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6789:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1243,
        GIR_Done,
      // Label 705: @21660
      GIM_Try, /*On fail goto*//*Label 706*/ 21724, // Rule ID 1245 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6788:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1245,
        GIR_Done,
      // Label 706: @21724
      GIM_Try, /*On fail goto*//*Label 707*/ 21788, // Rule ID 1247 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6787:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1247,
        GIR_Done,
      // Label 707: @21788
      GIM_Try, /*On fail goto*//*Label 708*/ 21852, // Rule ID 1249 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6786:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1249,
        GIR_Done,
      // Label 708: @21852
      GIM_Try, /*On fail goto*//*Label 709*/ 21916, // Rule ID 1251 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6785:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1251,
        GIR_Done,
      // Label 709: @21916
      GIM_Try, /*On fail goto*//*Label 710*/ 21980, // Rule ID 1253 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6784:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1253,
        GIR_Done,
      // Label 710: @21980
      GIM_Try, /*On fail goto*//*Label 711*/ 22044, // Rule ID 1255 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6783:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1255,
        GIR_Done,
      // Label 711: @22044
      GIM_Try, /*On fail goto*//*Label 712*/ 22108, // Rule ID 1257 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6782:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1257,
        GIR_Done,
      // Label 712: @22108
      GIM_Try, /*On fail goto*//*Label 713*/ 22172, // Rule ID 1259 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6781:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1259,
        GIR_Done,
      // Label 713: @22172
      GIM_Try, /*On fail goto*//*Label 714*/ 22236, // Rule ID 1261 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6780:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1261,
        GIR_Done,
      // Label 714: @22236
      GIM_Try, /*On fail goto*//*Label 715*/ 22300, // Rule ID 1263 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6779:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1263,
        GIR_Done,
      // Label 715: @22300
      GIM_Try, /*On fail goto*//*Label 716*/ 22364, // Rule ID 1265 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6778:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1265,
        GIR_Done,
      // Label 716: @22364
      GIM_Try, /*On fail goto*//*Label 717*/ 22428, // Rule ID 2585 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvpd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6645:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)  =>  (VBLENDVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2585,
        GIR_Done,
      // Label 717: @22428
      GIM_Try, /*On fail goto*//*Label 718*/ 22492, // Rule ID 2587 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 5499:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)  =>  (VBLENDVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2587,
        GIR_Done,
      // Label 718: @22492
      GIM_Try, /*On fail goto*//*Label 719*/ 22556, // Rule ID 2589 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6646:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)  =>  (VBLENDVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2589,
        GIR_Done,
      // Label 719: @22556
      GIM_Try, /*On fail goto*//*Label 720*/ 22620, // Rule ID 2591 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5500:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)  =>  (VBLENDVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2591,
        GIR_Done,
      // Label 720: @22620
      GIM_Try, /*On fail goto*//*Label 721*/ 22684, // Rule ID 2593 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6652:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)  =>  (VPBLENDVBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2593,
        GIR_Done,
      // Label 721: @22684
      GIM_Try, /*On fail goto*//*Label 722*/ 22748, // Rule ID 2595 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5588:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)  =>  (VPBLENDVBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2595,
        GIR_Done,
      // Label 722: @22748
      GIM_Reject,
    // Label 685: @22749
    GIM_Reject,
    // Label 8: @22750
    GIM_Try, /*On fail goto*//*Label 723*/ 22768, // Rule ID 1821 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_pause,
      // (intrinsic_void 6602:{ *:[iPTR] })  =>  (PAUSE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PAUSE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1821,
      GIR_Done,
    // Label 723: @22768
    GIM_Try, /*On fail goto*//*Label 724*/ 22788, // Rule ID 1822 //
      GIM_CheckFeatures, GIFBS_HasSSE1,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse_sfence,
      // (intrinsic_void 6558:{ *:[iPTR] })  =>  (SFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1822,
      GIR_Done,
    // Label 724: @22788
    GIM_Try, /*On fail goto*//*Label 725*/ 22808, // Rule ID 1823 //
      GIM_CheckFeatures, GIFBS_HasSSE2,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_lfence,
      // (intrinsic_void 6587:{ *:[iPTR] })  =>  (LFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1823,
      GIR_Done,
    // Label 725: @22808
    GIM_Try, /*On fail goto*//*Label 726*/ 22828, // Rule ID 1824 //
      GIM_CheckFeatures, GIFBS_HasMFence,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_mfence,
      // (intrinsic_void 6591:{ *:[iPTR] })  =>  (MFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1824,
      GIR_Done,
    // Label 726: @22828
    GIM_Try, /*On fail goto*//*Label 727*/ 22848, // Rule ID 2705 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroall,
      // (intrinsic_void 5553:{ *:[iPTR] })  =>  (VZEROALL)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROALL,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2705,
      GIR_Done,
    // Label 727: @22848
    GIM_Try, /*On fail goto*//*Label 728*/ 22868, // Rule ID 2706 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroupper,
      // (intrinsic_void 5554:{ *:[iPTR] })  =>  (VZEROUPPER)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROUPPER,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2706,
      GIR_Done,
    // Label 728: @22868
    GIM_Try, /*On fail goto*//*Label 729*/ 22888, // Rule ID 11956 //
      GIM_CheckFeatures, GIFBS_HasMMX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_emms,
      // (intrinsic_void 6418:{ *:[iPTR] })  =>  (MMX_EMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MMX_EMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11956,
      GIR_Done,
    // Label 729: @22888
    GIM_Try, /*On fail goto*//*Label 730*/ 22908, // Rule ID 12175 //
      GIM_CheckFeatures, GIFBS_Has3DNow,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_femms,
      // (intrinsic_void 6419:{ *:[iPTR] })  =>  (FEMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FEMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12175,
      GIR_Done,
    // Label 730: @22908
    GIM_Try, /*On fail goto*//*Label 731*/ 22928, // Rule ID 12190 //
      GIM_CheckFeatures, GIFBS_HasRTM,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xend,
      // (intrinsic_void 6743:{ *:[iPTR] })  =>  (XEND)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XEND,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12190,
      GIR_Done,
    // Label 731: @22928
    GIM_Try, /*On fail goto*//*Label 732*/ 22946, // Rule ID 12199 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbinvd,
      // (intrinsic_void 6730:{ *:[iPTR] })  =>  (WBINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12199,
      GIR_Done,
    // Label 732: @22946
    GIM_Try, /*On fail goto*//*Label 733*/ 22966, // Rule ID 12200 //
      GIM_CheckFeatures, GIFBS_HasWBNOINVD,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbnoinvd,
      // (intrinsic_void 6731:{ *:[iPTR] })  =>  (WBNOINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBNOINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12200,
      GIR_Done,
    // Label 733: @22966
    GIM_Try, /*On fail goto*//*Label 734*/ 22984, // Rule ID 12205 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_saveprevssp,
      // (intrinsic_void 6515:{ *:[iPTR] })  =>  (SAVEPREVSSP)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAVEPREVSSP,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12205,
      GIR_Done,
    // Label 734: @22984
    GIM_Try, /*On fail goto*//*Label 735*/ 23002, // Rule ID 12211 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_setssbsy,
      // (intrinsic_void 6520:{ *:[iPTR] })  =>  (SETSSBSY)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SETSSBSY,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12211,
      GIR_Done,
    // Label 735: @23002
    GIM_Try, /*On fail goto*//*Label 736*/ 23962,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
      GIM_Try, /*On fail goto*//*Label 737*/ 23033, // Rule ID 12196 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/1, 3,
        // (intrinsic_void 6411:{ *:[iPTR] }, 3:{ *:[i8] })  =>  (INT3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT3,
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12196,
        GIR_Done,
      // Label 737: @23033
      GIM_Try, /*On fail goto*//*Label 738*/ 23070, // Rule ID 12192 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xabort,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6741:{ *:[iPTR] }, (imm:{ *:[i8] }):$imm)  =>  (XABORT (imm:{ *:[i8] }):$imm)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XABORT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // imm
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12192,
        GIR_Done,
      // Label 738: @23070
      GIM_Try, /*On fail goto*//*Label 739*/ 23105, // Rule ID 12197 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6411:{ *:[iPTR] }, (imm:{ *:[i8] }):$trap)  =>  (INT (imm:{ *:[i8] }):$trap)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // trap
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12197,
        GIR_Done,
      // Label 739: @23105
      GIM_Try, /*On fail goto*//*Label 740*/ 23137, // Rule ID 1 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6389:{ *:[iPTR] })  =>  (RDFLAGS32:{ *:[i32] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1,
        GIR_Done,
      // Label 740: @23137
      GIM_Try, /*On fail goto*//*Label 741*/ 23169, // Rule ID 2 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6390:{ *:[iPTR] })  =>  (RDFLAGS64:{ *:[i64] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2,
        GIR_Done,
      // Label 741: @23169
      GIM_Try, /*On fail goto*//*Label 742*/ 23201, // Rule ID 95 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6528:{ *:[iPTR] })  =>  (SLWPCB:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 95,
        GIR_Done,
      // Label 742: @23201
      GIM_Try, /*On fail goto*//*Label 743*/ 23233, // Rule ID 97 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6528:{ *:[iPTR] })  =>  (SLWPCB64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 97,
        GIR_Done,
      // Label 743: @23233
      GIM_Try, /*On fail goto*//*Label 744*/ 23265, // Rule ID 12189 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xbegin,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6742:{ *:[iPTR] })  =>  (XBEGIN:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XBEGIN,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12189,
        GIR_Done,
      // Label 744: @23265
      GIM_Try, /*On fail goto*//*Label 745*/ 23295, // Rule ID 12227 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpkru,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6502:{ *:[iPTR] })  =>  (RDPKRU:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12227,
        GIR_Done,
      // Label 745: @23295
      GIM_Try, /*On fail goto*//*Label 746*/ 23327, // Rule ID 12228 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6497:{ *:[iPTR] })  =>  (RDFSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12228,
        GIR_Done,
      // Label 746: @23327
      GIM_Try, /*On fail goto*//*Label 747*/ 23359, // Rule ID 12229 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6498:{ *:[iPTR] })  =>  (RDFSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12229,
        GIR_Done,
      // Label 747: @23359
      GIM_Try, /*On fail goto*//*Label 748*/ 23391, // Rule ID 12230 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6499:{ *:[iPTR] })  =>  (RDGSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12230,
        GIR_Done,
      // Label 748: @23391
      GIM_Try, /*On fail goto*//*Label 749*/ 23423, // Rule ID 12231 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6500:{ *:[iPTR] })  =>  (RDGSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12231,
        GIR_Done,
      // Label 749: @23423
      GIM_Try, /*On fail goto*//*Label 750*/ 23455, // Rule ID 12237 //
        GIM_CheckFeatures, GIFBS_HasRDPID_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpid,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6501:{ *:[iPTR] })  =>  (RDPID32:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPID32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12237,
        GIR_Done,
      // Label 750: @23455
      GIM_Try, /*On fail goto*//*Label 751*/ 23487, // Rule ID 3 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6391:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFLAGS32:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3,
        GIR_Done,
      // Label 751: @23487
      GIM_Try, /*On fail goto*//*Label 752*/ 23519, // Rule ID 4 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6392:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFLAGS64:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4,
        GIR_Done,
      // Label 752: @23519
      GIM_Try, /*On fail goto*//*Label 753*/ 23551, // Rule ID 94 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6413:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (LLWPCB GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 94,
        GIR_Done,
      // Label 753: @23551
      GIM_Try, /*On fail goto*//*Label 754*/ 23583, // Rule ID 96 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6413:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (LLWPCB64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 96,
        GIR_Done,
      // Label 754: @23583
      GIM_Try, /*On fail goto*//*Label 755*/ 23615, // Rule ID 108 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
        // (intrinsic_void 6715:{ *:[iPTR] }, GR16:{ *:[i16] }:$src)  =>  (UMONITOR16 GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 108,
        GIR_Done,
      // Label 755: @23615
      GIM_Try, /*On fail goto*//*Label 756*/ 23647, // Rule ID 109 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6715:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (UMONITOR32 GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 109,
        GIR_Done,
      // Label 756: @23647
      GIM_Try, /*On fail goto*//*Label 757*/ 23679, // Rule ID 110 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6715:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (UMONITOR64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 110,
        GIR_Done,
      // Label 757: @23679
      GIM_Try, /*On fail goto*//*Label 758*/ 23709, // Rule ID 12201 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspd,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6409:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (INCSSPD GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12201,
        GIR_Done,
      // Label 758: @23709
      GIM_Try, /*On fail goto*//*Label 759*/ 23739, // Rule ID 12202 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspq,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6410:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (INCSSPQ GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12202,
        GIR_Done,
      // Label 759: @23739
      GIM_Try, /*On fail goto*//*Label 760*/ 23769, // Rule ID 12226 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrpkru,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6736:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRPKRU GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12226,
        GIR_Done,
      // Label 760: @23769
      GIM_Try, /*On fail goto*//*Label 761*/ 23801, // Rule ID 12232 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6732:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12232,
        GIR_Done,
      // Label 761: @23801
      GIM_Try, /*On fail goto*//*Label 762*/ 23833, // Rule ID 12233 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6733:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12233,
        GIR_Done,
      // Label 762: @23833
      GIM_Try, /*On fail goto*//*Label 763*/ 23865, // Rule ID 12234 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6734:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRGSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12234,
        GIR_Done,
      // Label 763: @23865
      GIM_Try, /*On fail goto*//*Label 764*/ 23897, // Rule ID 12235 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6735:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRGSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12235,
        GIR_Done,
      // Label 764: @23897
      GIM_Try, /*On fail goto*//*Label 765*/ 23929, // Rule ID 12240 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6495:{ *:[iPTR] }, GR32:{ *:[i32] }:$dst)  =>  (PTWRITEr GR32:{ *:[i32] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITEr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12240,
        GIR_Done,
      // Label 765: @23929
      GIM_Try, /*On fail goto*//*Label 766*/ 23961, // Rule ID 12241 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6496:{ *:[iPTR] }, GR64:{ *:[i64] }:$dst)  =>  (PTWRITE64r GR64:{ *:[i64] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITE64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12241,
        GIR_Done,
      // Label 766: @23961
      GIM_Reject,
    // Label 736: @23962
    GIM_Try, /*On fail goto*//*Label 767*/ 24052,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 768*/ 24009, // Rule ID 12203 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6510:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (RDSSPD:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12203,
        GIR_Done,
      // Label 768: @24009
      GIM_Try, /*On fail goto*//*Label 769*/ 24051, // Rule ID 12204 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6511:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (RDSSPQ:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12204,
        GIR_Done,
      // Label 769: @24051
      GIM_Reject,
    // Label 767: @24052
    GIM_Try, /*On fail goto*//*Label 770*/ 24180,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 771*/ 24118, // Rule ID 102 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6416:{ *:[iPTR] }, GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL32rri GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 102,
        GIR_Done,
      // Label 771: @24118
      GIM_Try, /*On fail goto*//*Label 772*/ 24179, // Rule ID 104 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6417:{ *:[iPTR] }, GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL64rri GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL64rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 104,
        GIR_Done,
      // Label 772: @24179
      GIM_Reject,
    // Label 770: @24180
    GIM_Reject,
    // Label 9: @24181
    GIM_Try, /*On fail goto*//*Label 773*/ 24296,
      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/1, 3, /*)*//*default:*//*Label 776*/ 24295,
      /*GILLT_s8*//*Label 774*/ 24195,
      /*GILLT_s16*//*Label 775*/ 24253,
      // Label 774: @24195
      GIM_Try, /*On fail goto*//*Label 777*/ 24252,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_Try, /*On fail goto*//*Label 778*/ 24238, // Rule ID 13874 //
          GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
          GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
          GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
          GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
          GIM_CheckIsSafeToFold, /*InsnID*/1,
          // (anyext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK8:{ *:[v8i1] }:$src, GR32:{ *:[i32] })
          GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
          GIR_EraseFromParent, /*InsnID*/0,
          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/26,
          // GIR_Coverage, 13874,
          GIR_Done,
        // Label 778: @24238
        GIM_Try, /*On fail goto*//*Label 779*/ 24251, // Rule ID 15656 //
          GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
          // (anyext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
          GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 15656,
          GIR_Done,
        // Label 779: @24251
        GIM_Reject,
      // Label 777: @24252
      GIM_Reject,
      // Label 775: @24253
      GIM_Try, /*On fail goto*//*Label 780*/ 24294, // Rule ID 13872 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (anyext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK16:{ *:[v16i1] }:$src, GR32:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/26,
        // GIR_Coverage, 13872,
        GIR_Done,
      // Label 780: @24294
      GIM_Reject,
      // Label 776: @24295
      GIM_Reject,
    // Label 773: @24296
    GIM_Reject,
    // Label 10: @24297
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 22, /*)*//*default:*//*Label 789*/ 24631,
    /*GILLT_s8*//*Label 781*/ 24324,
    /*GILLT_s16*//*Label 782*/ 24403, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 783*/ 24441, 0, 0,
    /*GILLT_v8s16*//*Label 784*/ 24465,
    /*GILLT_v8s32*//*Label 785*/ 24512, 0, 0,
    /*GILLT_v16s8*//*Label 786*/ 24536,
    /*GILLT_v16s16*//*Label 787*/ 24583, 0, 0,
    /*GILLT_v32s8*//*Label 788*/ 24607,
    // Label 781: @24324
    GIM_Try, /*On fail goto*//*Label 790*/ 24363, // Rule ID 15697 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i8] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/26,
      // GIR_Coverage, 15697,
      GIR_Done,
    // Label 790: @24363
    GIM_Try, /*On fail goto*//*Label 791*/ 24402, // Rule ID 15698 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (trunc:{ *:[i8] } GR16:{ *:[i16] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR16:{ *:[i16] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR16*/4,
      // GIR_Coverage, 15698,
      GIR_Done,
    // Label 791: @24402
    GIM_Reject,
    // Label 782: @24403
    GIM_Try, /*On fail goto*//*Label 792*/ 24440, // Rule ID 15691 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i16] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src, sub_16bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/3, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR16*/4,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/26,
      // GIR_Coverage, 15691,
      GIR_Done,
    // Label 792: @24440
    GIM_Reject,
    // Label 783: @24441
    GIM_Try, /*On fail goto*//*Label 793*/ 24464, // Rule ID 9585 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPMOVQDZ256rr:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9585,
      GIR_Done,
    // Label 793: @24464
    GIM_Reject,
    // Label 784: @24465
    GIM_Try, /*On fail goto*//*Label 794*/ 24488, // Rule ID 9561 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQWZrr:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9561,
      GIR_Done,
    // Label 794: @24488
    GIM_Try, /*On fail goto*//*Label 795*/ 24511, // Rule ID 9639 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9639,
      GIR_Done,
    // Label 795: @24511
    GIM_Reject,
    // Label 785: @24512
    GIM_Try, /*On fail goto*//*Label 796*/ 24535, // Rule ID 9588 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQDZrr:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9588,
      GIR_Done,
    // Label 796: @24535
    GIM_Reject,
    // Label 786: @24536
    GIM_Try, /*On fail goto*//*Label 797*/ 24559, // Rule ID 9615 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDBZrr:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9615,
      GIR_Done,
    // Label 797: @24559
    GIM_Try, /*On fail goto*//*Label 798*/ 24582, // Rule ID 9666 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)  =>  (VPMOVWBZ256rr:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9666,
      GIR_Done,
    // Label 798: @24582
    GIM_Reject,
    // Label 787: @24583
    GIM_Try, /*On fail goto*//*Label 799*/ 24606, // Rule ID 9642 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDWZrr:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9642,
      GIR_Done,
    // Label 799: @24606
    GIM_Reject,
    // Label 788: @24607
    GIM_Try, /*On fail goto*//*Label 800*/ 24630, // Rule ID 9669 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)  =>  (VPMOVWBZrr:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9669,
      GIR_Done,
    // Label 800: @24630
    GIM_Reject,
    // Label 789: @24631
    GIM_Reject,
    // Label 11: @24632
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 5, /*)*//*default:*//*Label 805*/ 24834,
    /*GILLT_s8*//*Label 801*/ 24642,
    /*GILLT_s16*//*Label 802*/ 24664,
    /*GILLT_s32*//*Label 803*/ 24686,
    /*GILLT_s64*//*Label 804*/ 24783,
    // Label 801: @24642
    GIM_Try, /*On fail goto*//*Label 806*/ 24663, // Rule ID 19 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i8] }):$src  =>  (MOV8ri:{ *:[i8] } (imm:{ *:[i8] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV8ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 19,
      GIR_Done,
    // Label 806: @24663
    GIM_Reject,
    // Label 802: @24664
    GIM_Try, /*On fail goto*//*Label 807*/ 24685, // Rule ID 20 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i16] }):$src  =>  (MOV16ri:{ *:[i16] } (imm:{ *:[i16] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 20,
      GIR_Done,
    // Label 807: @24685
    GIM_Reject,
    // Label 803: @24686
    GIM_Try, /*On fail goto*//*Label 808*/ 24708, // Rule ID 12259 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 0,
      // 0:{ *:[i32] }  =>  (MOV32r0:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12259,
      GIR_Done,
    // Label 808: @24708
    GIM_Try, /*On fail goto*//*Label 809*/ 24732, // Rule ID 12260 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
      // 1:{ *:[i32] }  =>  (MOV32r1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12260,
      GIR_Done,
    // Label 809: @24732
    GIM_Try, /*On fail goto*//*Label 810*/ 24756, // Rule ID 12261 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, -1,
      // -1:{ *:[i32] }  =>  (MOV32r_1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r_1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12261,
      GIR_Done,
    // Label 810: @24756
    GIM_Try, /*On fail goto*//*Label 811*/ 24782, // Rule ID 12262 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src  =>  (MOV32ImmSExti8:{ *:[i32] } (imm:{ *:[i32] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12262,
      GIR_Done,
    // Label 811: @24782
    GIM_Reject,
    // Label 804: @24783
    GIM_Try, /*On fail goto*//*Label 812*/ 24809, // Rule ID 12263 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src  =>  (MOV64ImmSExti8:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12263,
      GIR_Done,
    // Label 812: @24809
    GIM_Try, /*On fail goto*//*Label 813*/ 24833, // Rule ID 22 //
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src  =>  (MOV64ri32:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ri32,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 22,
      GIR_Done,
    // Label 813: @24833
    GIM_Reject,
    // Label 805: @24834
    GIM_Reject,
    // Label 12: @24835
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 817*/ 25027,
    /*GILLT_s32*//*Label 814*/ 24844,
    /*GILLT_s64*//*Label 815*/ 24891,
    /*GILLT_s80*//*Label 816*/ 24984,
    // Label 814: @24844
    GIM_Try, /*On fail goto*//*Label 818*/ 24867, // Rule ID 891 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm0>>  =>  (LD_Fp032:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp032,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 891,
      GIR_Done,
    // Label 818: @24867
    GIM_Try, /*On fail goto*//*Label 819*/ 24890, // Rule ID 892 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm1>>  =>  (LD_Fp132:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp132,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 892,
      GIR_Done,
    // Label 819: @24890
    GIM_Reject,
    // Label 815: @24891
    GIM_Try, /*On fail goto*//*Label 820*/ 24914, // Rule ID 893 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (LD_Fp064:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp064,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 893,
      GIR_Done,
    // Label 820: @24914
    GIM_Try, /*On fail goto*//*Label 821*/ 24937, // Rule ID 894 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm1>>  =>  (LD_Fp164:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp164,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 894,
      GIR_Done,
    // Label 821: @24937
    GIM_Try, /*On fail goto*//*Label 822*/ 24960, // Rule ID 1305 //
      GIM_CheckFeatures, GIFBS_HasSSE2_NoAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1305,
      GIR_Done,
    // Label 822: @24960
    GIM_Try, /*On fail goto*//*Label 823*/ 24983, // Rule ID 2812 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (AVX512_FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AVX512_FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2812,
      GIR_Done,
    // Label 823: @24983
    GIM_Reject,
    // Label 816: @24984
    GIM_Try, /*On fail goto*//*Label 824*/ 25005, // Rule ID 895 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm0>>  =>  (LD_Fp080:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp080,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 895,
      GIR_Done,
    // Label 824: @25005
    GIM_Try, /*On fail goto*//*Label 825*/ 25026, // Rule ID 896 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm1>>  =>  (LD_Fp180:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp180,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 896,
      GIR_Done,
    // Label 825: @25026
    GIM_Reject,
    // Label 817: @25027
    GIM_Reject,
    // Label 13: @25028
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 25, /*)*//*default:*//*Label 840*/ 25531,
    /*GILLT_s32*//*Label 826*/ 25056,
    /*GILLT_s64*//*Label 827*/ 25099, 0, 0, 0,
    /*GILLT_v2s64*//*Label 828*/ 25165, 0,
    /*GILLT_v4s32*//*Label 829*/ 25189,
    /*GILLT_v4s64*//*Label 830*/ 25213, 0,
    /*GILLT_v8s16*//*Label 831*/ 25237,
    /*GILLT_v8s32*//*Label 832*/ 25261,
    /*GILLT_v8s64*//*Label 833*/ 25285, 0,
    /*GILLT_v16s8*//*Label 834*/ 25309,
    /*GILLT_v16s16*//*Label 835*/ 25372,
    /*GILLT_v16s32*//*Label 836*/ 25435, 0,
    /*GILLT_v32s8*//*Label 837*/ 25459,
    /*GILLT_v32s16*//*Label 838*/ 25483, 0,
    /*GILLT_v64s8*//*Label 839*/ 25507,
    // Label 826: @25056
    GIM_Try, /*On fail goto*//*Label 841*/ 25077, // Rule ID 536 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVSX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 536,
      GIR_Done,
    // Label 841: @25077
    GIM_Try, /*On fail goto*//*Label 842*/ 25098, // Rule ID 538 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVSX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 538,
      GIR_Done,
    // Label 842: @25098
    GIM_Reject,
    // Label 827: @25099
    GIM_Try, /*On fail goto*//*Label 843*/ 25120, // Rule ID 544 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i64] } GR8:{ *:[i8] }:$src)  =>  (MOVSX64rr8:{ *:[i64] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 544,
      GIR_Done,
    // Label 843: @25120
    GIM_Try, /*On fail goto*//*Label 844*/ 25141, // Rule ID 546 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i64] } GR16:{ *:[i16] }:$src)  =>  (MOVSX64rr16:{ *:[i64] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 546,
      GIR_Done,
    // Label 844: @25141
    GIM_Try, /*On fail goto*//*Label 845*/ 25164, // Rule ID 548 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sext:{ *:[i64] } GR32:{ *:[i32] }:$src)  =>  (MOVSX64rr32:{ *:[i64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr32,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 548,
      GIR_Done,
    // Label 845: @25164
    GIM_Reject,
    // Label 828: @25165
    GIM_Try, /*On fail goto*//*Label 846*/ 25188, // Rule ID 9965 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      // (sext:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)  =>  (VPMOVM2QZ128rr:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9965,
      GIR_Done,
    // Label 846: @25188
    GIM_Reject,
    // Label 829: @25189
    GIM_Try, /*On fail goto*//*Label 847*/ 25212, // Rule ID 9962 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2DZ128rr:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9962,
      GIR_Done,
    // Label 847: @25212
    GIM_Reject,
    // Label 830: @25213
    GIM_Try, /*On fail goto*//*Label 848*/ 25236, // Rule ID 9964 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2QZ256rr:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9964,
      GIR_Done,
    // Label 848: @25236
    GIM_Reject,
    // Label 831: @25237
    GIM_Try, /*On fail goto*//*Label 849*/ 25260, // Rule ID 9959 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2WZ128rr:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9959,
      GIR_Done,
    // Label 849: @25260
    GIM_Reject,
    // Label 832: @25261
    GIM_Try, /*On fail goto*//*Label 850*/ 25284, // Rule ID 9961 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9961,
      GIR_Done,
    // Label 850: @25284
    GIM_Reject,
    // Label 833: @25285
    GIM_Try, /*On fail goto*//*Label 851*/ 25308, // Rule ID 9963 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2QZrr:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9963,
      GIR_Done,
    // Label 851: @25308
    GIM_Reject,
    // Label 834: @25309
    GIM_Try, /*On fail goto*//*Label 852*/ 25371,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 853*/ 25334, // Rule ID 9956 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2BZ128rr:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9956,
        GIR_Done,
      // Label 853: @25334
      GIM_Try, /*On fail goto*//*Label 854*/ 25370, // Rule ID 15189 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15189,
        GIR_Done,
      // Label 854: @25370
      GIM_Reject,
    // Label 852: @25371
    GIM_Reject,
    // Label 835: @25372
    GIM_Try, /*On fail goto*//*Label 855*/ 25434,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 856*/ 25397, // Rule ID 9958 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2WZ256rr:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9958,
        GIR_Done,
      // Label 856: @25397
      GIM_Try, /*On fail goto*//*Label 857*/ 25433, // Rule ID 15190 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDWZrr:{ *:[v16i16] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15190,
        GIR_Done,
      // Label 857: @25433
      GIM_Reject,
    // Label 855: @25434
    GIM_Reject,
    // Label 836: @25435
    GIM_Try, /*On fail goto*//*Label 858*/ 25458, // Rule ID 9960 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (sext:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9960,
      GIR_Done,
    // Label 858: @25458
    GIM_Reject,
    // Label 837: @25459
    GIM_Try, /*On fail goto*//*Label 859*/ 25482, // Rule ID 9955 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2BZ256rr:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9955,
      GIR_Done,
    // Label 859: @25482
    GIM_Reject,
    // Label 838: @25483
    GIM_Try, /*On fail goto*//*Label 860*/ 25506, // Rule ID 9957 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2WZrr:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9957,
      GIR_Done,
    // Label 860: @25506
    GIM_Reject,
    // Label 839: @25507
    GIM_Try, /*On fail goto*//*Label 861*/ 25530, // Rule ID 9954 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (sext:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)  =>  (VPMOVM2BZrr:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9954,
      GIR_Done,
    // Label 861: @25530
    GIM_Reject,
    // Label 840: @25531
    GIM_Reject,
    // Label 14: @25532
    GIM_Try, /*On fail goto*//*Label 862*/ 25666,
      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/1, 3, /*)*//*default:*//*Label 865*/ 25644,
      /*GILLT_s8*//*Label 863*/ 25546,
      /*GILLT_s16*//*Label 864*/ 25604,
      // Label 863: @25546
      GIM_Try, /*On fail goto*//*Label 866*/ 25603,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_Try, /*On fail goto*//*Label 867*/ 25589, // Rule ID 13873 //
          GIM_CheckFeatures, GIFBS_HasDQI,
          GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
          GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
          GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
          GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
          GIM_CheckIsSafeToFold, /*InsnID*/1,
          // (zext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (KMOVBrk:{ *:[i32] } VK8:{ *:[v8i1] }:$src)
          GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVBrk,
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
          GIR_EraseFromParent, /*InsnID*/0,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 13873,
          GIR_Done,
        // Label 867: @25589
        GIM_Try, /*On fail goto*//*Label 868*/ 25602, // Rule ID 540 //
          GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
          // (zext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
          GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 540,
          GIR_Done,
        // Label 868: @25602
        GIM_Reject,
      // Label 866: @25603
      GIM_Reject,
      // Label 864: @25604
      GIM_Try, /*On fail goto*//*Label 869*/ 25643, // Rule ID 13871 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (zext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (KMOVWrk:{ *:[i32] } VK16:{ *:[v16i1] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVWrk,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 13871,
        GIR_Done,
      // Label 869: @25643
      GIM_Reject,
      // Label 865: @25644
      GIM_Try, /*On fail goto*//*Label 870*/ 25665, // Rule ID 542 //
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
        // (zext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVZX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr16,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 542,
        GIR_Done,
      // Label 870: @25665
      GIM_Reject,
    // Label 862: @25666
    GIM_Reject,
    // Label 15: @25667
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 884*/ 26391,
    /*GILLT_s8*//*Label 871*/ 25695,
    /*GILLT_s16*//*Label 872*/ 25771,
    /*GILLT_s32*//*Label 873*/ 25847,
    /*GILLT_s64*//*Label 874*/ 25923, 0, 0, 0,
    /*GILLT_v2s64*//*Label 875*/ 25999, 0,
    /*GILLT_v4s32*//*Label 876*/ 26057,
    /*GILLT_v4s64*//*Label 877*/ 26115, 0,
    /*GILLT_v8s16*//*Label 878*/ 26173,
    /*GILLT_v8s32*//*Label 879*/ 26205,
    /*GILLT_v8s64*//*Label 880*/ 26263, 0, 0,
    /*GILLT_v16s16*//*Label 881*/ 26295,
    /*GILLT_v16s32*//*Label 882*/ 26327, 0, 0,
    /*GILLT_v32s16*//*Label 883*/ 26359,
    // Label 871: @25695
    GIM_Try, /*On fail goto*//*Label 885*/ 25770,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 886*/ 25739, // Rule ID 15713 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15713,
        GIR_Done,
      // Label 886: @25739
      GIM_Try, /*On fail goto*//*Label 887*/ 25769, // Rule ID 599 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 599,
        GIR_Done,
      // Label 887: @25769
      GIM_Reject,
    // Label 885: @25770
    GIM_Reject,
    // Label 872: @25771
    GIM_Try, /*On fail goto*//*Label 888*/ 25846,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 889*/ 25815, // Rule ID 15714 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15714,
        GIR_Done,
      // Label 889: @25815
      GIM_Try, /*On fail goto*//*Label 890*/ 25845, // Rule ID 600 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 600,
        GIR_Done,
      // Label 890: @25845
      GIM_Reject,
    // Label 888: @25846
    GIM_Reject,
    // Label 873: @25847
    GIM_Try, /*On fail goto*//*Label 891*/ 25922,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 892*/ 25891, // Rule ID 15715 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15715,
        GIR_Done,
      // Label 892: @25891
      GIM_Try, /*On fail goto*//*Label 893*/ 25921, // Rule ID 601 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 601,
        GIR_Done,
      // Label 893: @25921
      GIM_Reject,
    // Label 891: @25922
    GIM_Reject,
    // Label 874: @25923
    GIM_Try, /*On fail goto*//*Label 894*/ 25998,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 895*/ 25967, // Rule ID 15716 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15716,
        GIR_Done,
      // Label 895: @25967
      GIM_Try, /*On fail goto*//*Label 896*/ 25997, // Rule ID 602 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 602,
        GIR_Done,
      // Label 896: @25997
      GIM_Reject,
    // Label 894: @25998
    GIM_Reject,
    // Label 875: @25999
    GIM_Try, /*On fail goto*//*Label 897*/ 26056,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 898*/ 26032, // Rule ID 2755 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSLLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2755,
        GIR_Done,
      // Label 898: @26032
      GIM_Try, /*On fail goto*//*Label 899*/ 26055, // Rule ID 6429 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSLLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6429,
        GIR_Done,
      // Label 899: @26055
      GIM_Reject,
    // Label 897: @26056
    GIM_Reject,
    // Label 876: @26057
    GIM_Try, /*On fail goto*//*Label 900*/ 26114,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 901*/ 26090, // Rule ID 2751 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSLLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2751,
        GIR_Done,
      // Label 901: @26090
      GIM_Try, /*On fail goto*//*Label 902*/ 26113, // Rule ID 6402 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSLLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6402,
        GIR_Done,
      // Label 902: @26113
      GIM_Reject,
    // Label 900: @26114
    GIM_Reject,
    // Label 877: @26115
    GIM_Try, /*On fail goto*//*Label 903*/ 26172,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 904*/ 26148, // Rule ID 2757 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSLLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2757,
        GIR_Done,
      // Label 904: @26148
      GIM_Try, /*On fail goto*//*Label 905*/ 26171, // Rule ID 6420 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSLLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6420,
        GIR_Done,
      // Label 905: @26171
      GIM_Reject,
    // Label 903: @26172
    GIM_Reject,
    // Label 878: @26173
    GIM_Try, /*On fail goto*//*Label 906*/ 26204, // Rule ID 6450 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (shl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSLLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6450,
      GIR_Done,
    // Label 906: @26204
    GIM_Reject,
    // Label 879: @26205
    GIM_Try, /*On fail goto*//*Label 907*/ 26262,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 908*/ 26238, // Rule ID 2753 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSLLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2753,
        GIR_Done,
      // Label 908: @26238
      GIM_Try, /*On fail goto*//*Label 909*/ 26261, // Rule ID 6393 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSLLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6393,
        GIR_Done,
      // Label 909: @26261
      GIM_Reject,
    // Label 907: @26262
    GIM_Reject,
    // Label 880: @26263
    GIM_Try, /*On fail goto*//*Label 910*/ 26294, // Rule ID 6411 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSLLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6411,
      GIR_Done,
    // Label 910: @26294
    GIM_Reject,
    // Label 881: @26295
    GIM_Try, /*On fail goto*//*Label 911*/ 26326, // Rule ID 6444 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (shl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSLLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6444,
      GIR_Done,
    // Label 911: @26326
    GIM_Reject,
    // Label 882: @26327
    GIM_Try, /*On fail goto*//*Label 912*/ 26358, // Rule ID 6384 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSLLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6384,
      GIR_Done,
    // Label 912: @26358
    GIM_Reject,
    // Label 883: @26359
    GIM_Try, /*On fail goto*//*Label 913*/ 26390, // Rule ID 6438 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSLLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6438,
      GIR_Done,
    // Label 913: @26390
    GIM_Reject,
    // Label 884: @26391
    GIM_Reject,
    // Label 16: @26392
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 927*/ 27222,
    /*GILLT_s8*//*Label 914*/ 26420,
    /*GILLT_s16*//*Label 915*/ 26492,
    /*GILLT_s32*//*Label 916*/ 26564,
    /*GILLT_s64*//*Label 917*/ 26758, 0, 0, 0,
    /*GILLT_v2s64*//*Label 918*/ 26830, 0,
    /*GILLT_v4s32*//*Label 919*/ 26888,
    /*GILLT_v4s64*//*Label 920*/ 26946, 0,
    /*GILLT_v8s16*//*Label 921*/ 27004,
    /*GILLT_v8s32*//*Label 922*/ 27036,
    /*GILLT_v8s64*//*Label 923*/ 27094, 0, 0,
    /*GILLT_v16s16*//*Label 924*/ 27126,
    /*GILLT_v16s32*//*Label 925*/ 27158, 0, 0,
    /*GILLT_v32s16*//*Label 926*/ 27190,
    // Label 914: @26420
    GIM_Try, /*On fail goto*//*Label 928*/ 26491,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 929*/ 26460, // Rule ID 623 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SHR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 623,
        GIR_Done,
      // Label 929: @26460
      GIM_Try, /*On fail goto*//*Label 930*/ 26490, // Rule ID 619 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 619,
        GIR_Done,
      // Label 930: @26490
      GIM_Reject,
    // Label 928: @26491
    GIM_Reject,
    // Label 915: @26492
    GIM_Try, /*On fail goto*//*Label 931*/ 26563,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 932*/ 26532, // Rule ID 624 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SHR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 624,
        GIR_Done,
      // Label 932: @26532
      GIM_Try, /*On fail goto*//*Label 933*/ 26562, // Rule ID 620 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 620,
        GIR_Done,
      // Label 933: @26562
      GIM_Reject,
    // Label 931: @26563
    GIM_Reject,
    // Label 916: @26564
    GIM_Try, /*On fail goto*//*Label 934*/ 26757,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 935*/ 26696, // Rule ID 16217 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SHL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/4, /*MI*/0, /*OpIdx*/2, // MIs[4]
        GIM_CheckOpcode, /*MI*/4, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/4, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/5, /*MI*/4, /*OpIdx*/1, // MIs[5]
        GIM_CheckOpcode, /*MI*/5, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/5, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/5, /*Op*/1, 32,
        // MIs[5] lz
        GIM_CheckIsSameOperand, /*MI*/5, /*OpIdx*/2, /*OtherMI*/3, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        GIM_CheckIsSafeToFold, /*InsnID*/4,
        GIM_CheckIsSafeToFold, /*InsnID*/5,
        // (srl:{ *:[i32] } (shl:{ *:[i32] } GR32:{ *:[i32] }:$src, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))), (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz)))  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16217,
        GIR_Done,
      // Label 935: @26696
      GIM_Try, /*On fail goto*//*Label 936*/ 26722, // Rule ID 625 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SHR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 625,
        GIR_Done,
      // Label 936: @26722
      GIM_Try, /*On fail goto*//*Label 937*/ 26756, // Rule ID 621 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 621,
        GIR_Done,
      // Label 937: @26756
      GIM_Reject,
    // Label 934: @26757
    GIM_Reject,
    // Label 917: @26758
    GIM_Try, /*On fail goto*//*Label 938*/ 26829,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 939*/ 26798, // Rule ID 626 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SHR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 626,
        GIR_Done,
      // Label 939: @26798
      GIM_Try, /*On fail goto*//*Label 940*/ 26828, // Rule ID 622 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 622,
        GIR_Done,
      // Label 940: @26828
      GIM_Reject,
    // Label 938: @26829
    GIM_Reject,
    // Label 918: @26830
    GIM_Try, /*On fail goto*//*Label 941*/ 26887,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 942*/ 26863, // Rule ID 2763 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSRLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2763,
        GIR_Done,
      // Label 942: @26863
      GIM_Try, /*On fail goto*//*Label 943*/ 26886, // Rule ID 6573 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6573,
        GIR_Done,
      // Label 943: @26886
      GIM_Reject,
    // Label 941: @26887
    GIM_Reject,
    // Label 919: @26888
    GIM_Try, /*On fail goto*//*Label 944*/ 26945,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 945*/ 26921, // Rule ID 2759 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2759,
        GIR_Done,
      // Label 945: @26921
      GIM_Try, /*On fail goto*//*Label 946*/ 26944, // Rule ID 6546 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6546,
        GIR_Done,
      // Label 946: @26944
      GIM_Reject,
    // Label 944: @26945
    GIM_Reject,
    // Label 920: @26946
    GIM_Try, /*On fail goto*//*Label 947*/ 27003,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 948*/ 26979, // Rule ID 2765 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSRLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2765,
        GIR_Done,
      // Label 948: @26979
      GIM_Try, /*On fail goto*//*Label 949*/ 27002, // Rule ID 6564 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6564,
        GIR_Done,
      // Label 949: @27002
      GIM_Reject,
    // Label 947: @27003
    GIM_Reject,
    // Label 921: @27004
    GIM_Try, /*On fail goto*//*Label 950*/ 27035, // Rule ID 6594 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (srl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6594,
      GIR_Done,
    // Label 950: @27035
    GIM_Reject,
    // Label 922: @27036
    GIM_Try, /*On fail goto*//*Label 951*/ 27093,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 952*/ 27069, // Rule ID 2761 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2761,
        GIR_Done,
      // Label 952: @27069
      GIM_Try, /*On fail goto*//*Label 953*/ 27092, // Rule ID 6537 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6537,
        GIR_Done,
      // Label 953: @27092
      GIM_Reject,
    // Label 951: @27093
    GIM_Reject,
    // Label 923: @27094
    GIM_Try, /*On fail goto*//*Label 954*/ 27125, // Rule ID 6555 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6555,
      GIR_Done,
    // Label 954: @27125
    GIM_Reject,
    // Label 924: @27126
    GIM_Try, /*On fail goto*//*Label 955*/ 27157, // Rule ID 6588 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (srl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6588,
      GIR_Done,
    // Label 955: @27157
    GIM_Reject,
    // Label 925: @27158
    GIM_Try, /*On fail goto*//*Label 956*/ 27189, // Rule ID 6528 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6528,
      GIR_Done,
    // Label 956: @27189
    GIM_Reject,
    // Label 926: @27190
    GIM_Try, /*On fail goto*//*Label 957*/ 27221, // Rule ID 6582 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6582,
      GIR_Done,
    // Label 957: @27221
    GIM_Reject,
    // Label 927: @27222
    GIM_Reject,
    // Label 17: @27223
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 971*/ 27879,
    /*GILLT_s8*//*Label 958*/ 27251,
    /*GILLT_s16*//*Label 959*/ 27323,
    /*GILLT_s32*//*Label 960*/ 27395,
    /*GILLT_s64*//*Label 961*/ 27467, 0, 0, 0,
    /*GILLT_v2s64*//*Label 962*/ 27539, 0,
    /*GILLT_v4s32*//*Label 963*/ 27571,
    /*GILLT_v4s64*//*Label 964*/ 27629, 0,
    /*GILLT_v8s16*//*Label 965*/ 27661,
    /*GILLT_v8s32*//*Label 966*/ 27693,
    /*GILLT_v8s64*//*Label 967*/ 27751, 0, 0,
    /*GILLT_v16s16*//*Label 968*/ 27783,
    /*GILLT_v16s32*//*Label 969*/ 27815, 0, 0,
    /*GILLT_v32s16*//*Label 970*/ 27847,
    // Label 958: @27251
    GIM_Try, /*On fail goto*//*Label 972*/ 27322,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 973*/ 27291, // Rule ID 647 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SAR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 647,
        GIR_Done,
      // Label 973: @27291
      GIM_Try, /*On fail goto*//*Label 974*/ 27321, // Rule ID 643 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 643,
        GIR_Done,
      // Label 974: @27321
      GIM_Reject,
    // Label 972: @27322
    GIM_Reject,
    // Label 959: @27323
    GIM_Try, /*On fail goto*//*Label 975*/ 27394,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 976*/ 27363, // Rule ID 648 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SAR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 648,
        GIR_Done,
      // Label 976: @27363
      GIM_Try, /*On fail goto*//*Label 977*/ 27393, // Rule ID 644 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 644,
        GIR_Done,
      // Label 977: @27393
      GIM_Reject,
    // Label 975: @27394
    GIM_Reject,
    // Label 960: @27395
    GIM_Try, /*On fail goto*//*Label 978*/ 27466,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 979*/ 27435, // Rule ID 649 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SAR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 649,
        GIR_Done,
      // Label 979: @27435
      GIM_Try, /*On fail goto*//*Label 980*/ 27465, // Rule ID 645 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 645,
        GIR_Done,
      // Label 980: @27465
      GIM_Reject,
    // Label 978: @27466
    GIM_Reject,
    // Label 961: @27467
    GIM_Try, /*On fail goto*//*Label 981*/ 27538,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 982*/ 27507, // Rule ID 650 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SAR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 650,
        GIR_Done,
      // Label 982: @27507
      GIM_Try, /*On fail goto*//*Label 983*/ 27537, // Rule ID 646 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 646,
        GIR_Done,
      // Label 983: @27537
      GIM_Reject,
    // Label 981: @27538
    GIM_Reject,
    // Label 962: @27539
    GIM_Try, /*On fail goto*//*Label 984*/ 27570, // Rule ID 6501 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRAVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6501,
      GIR_Done,
    // Label 984: @27570
    GIM_Reject,
    // Label 963: @27571
    GIM_Try, /*On fail goto*//*Label 985*/ 27628,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 986*/ 27604, // Rule ID 2767 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sra:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRAVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2767,
        GIR_Done,
      // Label 986: @27604
      GIM_Try, /*On fail goto*//*Label 987*/ 27627, // Rule ID 6474 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sra:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRAVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6474,
        GIR_Done,
      // Label 987: @27627
      GIM_Reject,
    // Label 985: @27628
    GIM_Reject,
    // Label 964: @27629
    GIM_Try, /*On fail goto*//*Label 988*/ 27660, // Rule ID 6492 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRAVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6492,
      GIR_Done,
    // Label 988: @27660
    GIM_Reject,
    // Label 965: @27661
    GIM_Try, /*On fail goto*//*Label 989*/ 27692, // Rule ID 6522 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRAVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6522,
      GIR_Done,
    // Label 989: @27692
    GIM_Reject,
    // Label 966: @27693
    GIM_Try, /*On fail goto*//*Label 990*/ 27750,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 991*/ 27726, // Rule ID 2769 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sra:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRAVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2769,
        GIR_Done,
      // Label 991: @27726
      GIM_Try, /*On fail goto*//*Label 992*/ 27749, // Rule ID 6465 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sra:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRAVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6465,
        GIR_Done,
      // Label 992: @27749
      GIM_Reject,
    // Label 990: @27750
    GIM_Reject,
    // Label 967: @27751
    GIM_Try, /*On fail goto*//*Label 993*/ 27782, // Rule ID 6483 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRAVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6483,
      GIR_Done,
    // Label 993: @27782
    GIM_Reject,
    // Label 968: @27783
    GIM_Try, /*On fail goto*//*Label 994*/ 27814, // Rule ID 6516 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRAVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6516,
      GIR_Done,
    // Label 994: @27814
    GIM_Reject,
    // Label 969: @27815
    GIM_Try, /*On fail goto*//*Label 995*/ 27846, // Rule ID 6456 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRAVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6456,
      GIR_Done,
    // Label 995: @27846
    GIM_Reject,
    // Label 970: @27847
    GIM_Try, /*On fail goto*//*Label 996*/ 27878, // Rule ID 6510 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRAVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6510,
      GIR_Done,
    // Label 996: @27878
    GIM_Reject,
    // Label 971: @27879
    GIM_Reject,
    // Label 18: @27880
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1006*/ 28492,
    /*GILLT_s32*//*Label 997*/ 27903,
    /*GILLT_s64*//*Label 998*/ 28010,
    /*GILLT_s80*//*Label 999*/ 28117, 0, 0,
    /*GILLT_v2s64*//*Label 1000*/ 28150, 0,
    /*GILLT_v4s32*//*Label 1001*/ 28231,
    /*GILLT_v4s64*//*Label 1002*/ 28312, 0, 0,
    /*GILLT_v8s32*//*Label 1003*/ 28370,
    /*GILLT_v8s64*//*Label 1004*/ 28428, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1005*/ 28460,
    // Label 997: @27903
    GIM_Try, /*On fail goto*//*Label 1007*/ 28009,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1008*/ 27939, // Rule ID 744 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fadd:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (ADD_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 744,
        GIR_Done,
      // Label 1008: @27939
      GIM_Try, /*On fail goto*//*Label 1009*/ 27962, // Rule ID 1609 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1609,
        GIR_Done,
      // Label 1009: @27962
      GIM_Try, /*On fail goto*//*Label 1010*/ 27985, // Rule ID 1613 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (ADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1613,
        GIR_Done,
      // Label 1010: @27985
      GIM_Try, /*On fail goto*//*Label 1011*/ 28008, // Rule ID 5098 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fadd:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VADDSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5098,
        GIR_Done,
      // Label 1011: @28008
      GIM_Reject,
    // Label 1007: @28009
    GIM_Reject,
    // Label 998: @28010
    GIM_Try, /*On fail goto*//*Label 1012*/ 28116,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1013*/ 28046, // Rule ID 745 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fadd:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (ADD_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 745,
        GIR_Done,
      // Label 1013: @28046
      GIM_Try, /*On fail goto*//*Label 1014*/ 28069, // Rule ID 1611 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1611,
        GIR_Done,
      // Label 1014: @28069
      GIM_Try, /*On fail goto*//*Label 1015*/ 28092, // Rule ID 1615 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (ADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1615,
        GIR_Done,
      // Label 1015: @28092
      GIM_Try, /*On fail goto*//*Label 1016*/ 28115, // Rule ID 5109 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fadd:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VADDSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5109,
        GIR_Done,
      // Label 1016: @28115
      GIM_Reject,
    // Label 1012: @28116
    GIM_Reject,
    // Label 999: @28117
    GIM_Try, /*On fail goto*//*Label 1017*/ 28149, // Rule ID 746 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fadd:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (ADD_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 746,
      GIR_Done,
    // Label 1017: @28149
    GIM_Reject,
    // Label 1000: @28150
    GIM_Try, /*On fail goto*//*Label 1018*/ 28230,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1019*/ 28183, // Rule ID 1599 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1599,
        GIR_Done,
      // Label 1019: @28183
      GIM_Try, /*On fail goto*//*Label 1020*/ 28206, // Rule ID 1607 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (ADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1607,
        GIR_Done,
      // Label 1020: @28206
      GIM_Try, /*On fail goto*//*Label 1021*/ 28229, // Rule ID 5268 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VADDPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5268,
        GIR_Done,
      // Label 1021: @28229
      GIM_Reject,
    // Label 1018: @28230
    GIM_Reject,
    // Label 1001: @28231
    GIM_Try, /*On fail goto*//*Label 1022*/ 28311,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1023*/ 28264, // Rule ID 1597 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1597,
        GIR_Done,
      // Label 1023: @28264
      GIM_Try, /*On fail goto*//*Label 1024*/ 28287, // Rule ID 1605 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (ADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1605,
        GIR_Done,
      // Label 1024: @28287
      GIM_Try, /*On fail goto*//*Label 1025*/ 28310, // Rule ID 5250 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VADDPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5250,
        GIR_Done,
      // Label 1025: @28310
      GIM_Reject,
    // Label 1022: @28311
    GIM_Reject,
    // Label 1002: @28312
    GIM_Try, /*On fail goto*//*Label 1026*/ 28369,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1027*/ 28345, // Rule ID 1603 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VADDPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1603,
        GIR_Done,
      // Label 1027: @28345
      GIM_Try, /*On fail goto*//*Label 1028*/ 28368, // Rule ID 5277 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VADDPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5277,
        GIR_Done,
      // Label 1028: @28368
      GIM_Reject,
    // Label 1026: @28369
    GIM_Reject,
    // Label 1003: @28370
    GIM_Try, /*On fail goto*//*Label 1029*/ 28427,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1030*/ 28403, // Rule ID 1601 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VADDPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1601,
        GIR_Done,
      // Label 1030: @28403
      GIM_Try, /*On fail goto*//*Label 1031*/ 28426, // Rule ID 5259 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VADDPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5259,
        GIR_Done,
      // Label 1031: @28426
      GIM_Reject,
    // Label 1029: @28427
    GIM_Reject,
    // Label 1004: @28428
    GIM_Try, /*On fail goto*//*Label 1032*/ 28459, // Rule ID 5241 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VADDPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5241,
      GIR_Done,
    // Label 1032: @28459
    GIM_Reject,
    // Label 1005: @28460
    GIM_Try, /*On fail goto*//*Label 1033*/ 28491, // Rule ID 5232 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VADDPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5232,
      GIR_Done,
    // Label 1033: @28491
    GIM_Reject,
    // Label 1006: @28492
    GIM_Reject,
    // Label 19: @28493
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1043*/ 29105,
    /*GILLT_s32*//*Label 1034*/ 28516,
    /*GILLT_s64*//*Label 1035*/ 28623,
    /*GILLT_s80*//*Label 1036*/ 28730, 0, 0,
    /*GILLT_v2s64*//*Label 1037*/ 28763, 0,
    /*GILLT_v4s32*//*Label 1038*/ 28844,
    /*GILLT_v4s64*//*Label 1039*/ 28925, 0, 0,
    /*GILLT_v8s32*//*Label 1040*/ 28983,
    /*GILLT_v8s64*//*Label 1041*/ 29041, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1042*/ 29073,
    // Label 1034: @28516
    GIM_Try, /*On fail goto*//*Label 1044*/ 28622,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1045*/ 28552, // Rule ID 747 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fsub:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (SUB_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 747,
        GIR_Done,
      // Label 1045: @28552
      GIM_Try, /*On fail goto*//*Label 1046*/ 28575, // Rule ID 1649 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VSUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1649,
        GIR_Done,
      // Label 1046: @28575
      GIM_Try, /*On fail goto*//*Label 1047*/ 28598, // Rule ID 1653 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (SUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1653,
        GIR_Done,
      // Label 1047: @28598
      GIM_Try, /*On fail goto*//*Label 1048*/ 28621, // Rule ID 5142 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fsub:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VSUBSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5142,
        GIR_Done,
      // Label 1048: @28621
      GIM_Reject,
    // Label 1044: @28622
    GIM_Reject,
    // Label 1035: @28623
    GIM_Try, /*On fail goto*//*Label 1049*/ 28729,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1050*/ 28659, // Rule ID 748 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fsub:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (SUB_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 748,
        GIR_Done,
      // Label 1050: @28659
      GIM_Try, /*On fail goto*//*Label 1051*/ 28682, // Rule ID 1651 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VSUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1651,
        GIR_Done,
      // Label 1051: @28682
      GIM_Try, /*On fail goto*//*Label 1052*/ 28705, // Rule ID 1655 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (SUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1655,
        GIR_Done,
      // Label 1052: @28705
      GIM_Try, /*On fail goto*//*Label 1053*/ 28728, // Rule ID 5153 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fsub:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VSUBSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5153,
        GIR_Done,
      // Label 1053: @28728
      GIM_Reject,
    // Label 1049: @28729
    GIM_Reject,
    // Label 1036: @28730
    GIM_Try, /*On fail goto*//*Label 1054*/ 28762, // Rule ID 749 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fsub:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (SUB_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 749,
      GIR_Done,
    // Label 1054: @28762
    GIM_Reject,
    // Label 1037: @28763
    GIM_Try, /*On fail goto*//*Label 1055*/ 28843,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1056*/ 28796, // Rule ID 1639 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VSUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1639,
        GIR_Done,
      // Label 1056: @28796
      GIM_Try, /*On fail goto*//*Label 1057*/ 28819, // Rule ID 1647 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (SUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1647,
        GIR_Done,
      // Label 1057: @28819
      GIM_Try, /*On fail goto*//*Label 1058*/ 28842, // Rule ID 5388 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VSUBPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5388,
        GIR_Done,
      // Label 1058: @28842
      GIM_Reject,
    // Label 1055: @28843
    GIM_Reject,
    // Label 1038: @28844
    GIM_Try, /*On fail goto*//*Label 1059*/ 28924,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1060*/ 28877, // Rule ID 1637 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VSUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1637,
        GIR_Done,
      // Label 1060: @28877
      GIM_Try, /*On fail goto*//*Label 1061*/ 28900, // Rule ID 1645 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (SUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1645,
        GIR_Done,
      // Label 1061: @28900
      GIM_Try, /*On fail goto*//*Label 1062*/ 28923, // Rule ID 5370 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VSUBPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5370,
        GIR_Done,
      // Label 1062: @28923
      GIM_Reject,
    // Label 1059: @28924
    GIM_Reject,
    // Label 1039: @28925
    GIM_Try, /*On fail goto*//*Label 1063*/ 28982,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1064*/ 28958, // Rule ID 1643 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VSUBPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1643,
        GIR_Done,
      // Label 1064: @28958
      GIM_Try, /*On fail goto*//*Label 1065*/ 28981, // Rule ID 5397 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VSUBPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5397,
        GIR_Done,
      // Label 1065: @28981
      GIM_Reject,
    // Label 1063: @28982
    GIM_Reject,
    // Label 1040: @28983
    GIM_Try, /*On fail goto*//*Label 1066*/ 29040,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1067*/ 29016, // Rule ID 1641 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VSUBPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1641,
        GIR_Done,
      // Label 1067: @29016
      GIM_Try, /*On fail goto*//*Label 1068*/ 29039, // Rule ID 5379 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VSUBPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5379,
        GIR_Done,
      // Label 1068: @29039
      GIM_Reject,
    // Label 1066: @29040
    GIM_Reject,
    // Label 1041: @29041
    GIM_Try, /*On fail goto*//*Label 1069*/ 29072, // Rule ID 5361 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VSUBPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5361,
      GIR_Done,
    // Label 1069: @29072
    GIM_Reject,
    // Label 1042: @29073
    GIM_Try, /*On fail goto*//*Label 1070*/ 29104, // Rule ID 5352 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VSUBPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5352,
      GIR_Done,
    // Label 1070: @29104
    GIM_Reject,
    // Label 1043: @29105
    GIM_Reject,
    // Label 20: @29106
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1080*/ 29718,
    /*GILLT_s32*//*Label 1071*/ 29129,
    /*GILLT_s64*//*Label 1072*/ 29236,
    /*GILLT_s80*//*Label 1073*/ 29343, 0, 0,
    /*GILLT_v2s64*//*Label 1074*/ 29376, 0,
    /*GILLT_v4s32*//*Label 1075*/ 29457,
    /*GILLT_v4s64*//*Label 1076*/ 29538, 0, 0,
    /*GILLT_v8s32*//*Label 1077*/ 29596,
    /*GILLT_v8s64*//*Label 1078*/ 29654, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1079*/ 29686,
    // Label 1071: @29129
    GIM_Try, /*On fail goto*//*Label 1081*/ 29235,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1082*/ 29165, // Rule ID 750 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fmul:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (MUL_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 750,
        GIR_Done,
      // Label 1082: @29165
      GIM_Try, /*On fail goto*//*Label 1083*/ 29188, // Rule ID 1629 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VMULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1629,
        GIR_Done,
      // Label 1083: @29188
      GIM_Try, /*On fail goto*//*Label 1084*/ 29211, // Rule ID 1633 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (MULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1633,
        GIR_Done,
      // Label 1084: @29211
      GIM_Try, /*On fail goto*//*Label 1085*/ 29234, // Rule ID 5120 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fmul:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VMULSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5120,
        GIR_Done,
      // Label 1085: @29234
      GIM_Reject,
    // Label 1081: @29235
    GIM_Reject,
    // Label 1072: @29236
    GIM_Try, /*On fail goto*//*Label 1086*/ 29342,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1087*/ 29272, // Rule ID 751 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fmul:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (MUL_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 751,
        GIR_Done,
      // Label 1087: @29272
      GIM_Try, /*On fail goto*//*Label 1088*/ 29295, // Rule ID 1631 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VMULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1631,
        GIR_Done,
      // Label 1088: @29295
      GIM_Try, /*On fail goto*//*Label 1089*/ 29318, // Rule ID 1635 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (MULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1635,
        GIR_Done,
      // Label 1089: @29318
      GIM_Try, /*On fail goto*//*Label 1090*/ 29341, // Rule ID 5131 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fmul:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VMULSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5131,
        GIR_Done,
      // Label 1090: @29341
      GIM_Reject,
    // Label 1086: @29342
    GIM_Reject,
    // Label 1073: @29343
    GIM_Try, /*On fail goto*//*Label 1091*/ 29375, // Rule ID 752 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fmul:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (MUL_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 752,
      GIR_Done,
    // Label 1091: @29375
    GIM_Reject,
    // Label 1074: @29376
    GIM_Try, /*On fail goto*//*Label 1092*/ 29456,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1093*/ 29409, // Rule ID 1619 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VMULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1619,
        GIR_Done,
      // Label 1093: @29409
      GIM_Try, /*On fail goto*//*Label 1094*/ 29432, // Rule ID 1627 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (MULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1627,
        GIR_Done,
      // Label 1094: @29432
      GIM_Try, /*On fail goto*//*Label 1095*/ 29455, // Rule ID 5328 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VMULPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5328,
        GIR_Done,
      // Label 1095: @29455
      GIM_Reject,
    // Label 1092: @29456
    GIM_Reject,
    // Label 1075: @29457
    GIM_Try, /*On fail goto*//*Label 1096*/ 29537,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1097*/ 29490, // Rule ID 1617 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VMULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1617,
        GIR_Done,
      // Label 1097: @29490
      GIM_Try, /*On fail goto*//*Label 1098*/ 29513, // Rule ID 1625 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (MULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1625,
        GIR_Done,
      // Label 1098: @29513
      GIM_Try, /*On fail goto*//*Label 1099*/ 29536, // Rule ID 5310 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VMULPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5310,
        GIR_Done,
      // Label 1099: @29536
      GIM_Reject,
    // Label 1096: @29537
    GIM_Reject,
    // Label 1076: @29538
    GIM_Try, /*On fail goto*//*Label 1100*/ 29595,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1101*/ 29571, // Rule ID 1623 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VMULPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1623,
        GIR_Done,
      // Label 1101: @29571
      GIM_Try, /*On fail goto*//*Label 1102*/ 29594, // Rule ID 5337 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VMULPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5337,
        GIR_Done,
      // Label 1102: @29594
      GIM_Reject,
    // Label 1100: @29595
    GIM_Reject,
    // Label 1077: @29596
    GIM_Try, /*On fail goto*//*Label 1103*/ 29653,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1104*/ 29629, // Rule ID 1621 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VMULPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1621,
        GIR_Done,
      // Label 1104: @29629
      GIM_Try, /*On fail goto*//*Label 1105*/ 29652, // Rule ID 5319 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VMULPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5319,
        GIR_Done,
      // Label 1105: @29652
      GIM_Reject,
    // Label 1103: @29653
    GIM_Reject,
    // Label 1078: @29654
    GIM_Try, /*On fail goto*//*Label 1106*/ 29685, // Rule ID 5301 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VMULPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5301,
      GIR_Done,
    // Label 1106: @29685
    GIM_Reject,
    // Label 1079: @29686
    GIM_Try, /*On fail goto*//*Label 1107*/ 29717, // Rule ID 5292 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VMULPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5292,
      GIR_Done,
    // Label 1107: @29717
    GIM_Reject,
    // Label 1080: @29718
    GIM_Reject,
    // Label 21: @29719
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1117*/ 30331,
    /*GILLT_s32*//*Label 1108*/ 29742,
    /*GILLT_s64*//*Label 1109*/ 29849,
    /*GILLT_s80*//*Label 1110*/ 29956, 0, 0,
    /*GILLT_v2s64*//*Label 1111*/ 29989, 0,
    /*GILLT_v4s32*//*Label 1112*/ 30070,
    /*GILLT_v4s64*//*Label 1113*/ 30151, 0, 0,
    /*GILLT_v8s32*//*Label 1114*/ 30209,
    /*GILLT_v8s64*//*Label 1115*/ 30267, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1116*/ 30299,
    // Label 1108: @29742
    GIM_Try, /*On fail goto*//*Label 1118*/ 29848,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1119*/ 29778, // Rule ID 753 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fdiv:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (DIV_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 753,
        GIR_Done,
      // Label 1119: @29778
      GIM_Try, /*On fail goto*//*Label 1120*/ 29801, // Rule ID 1669 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VDIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1669,
        GIR_Done,
      // Label 1120: @29801
      GIM_Try, /*On fail goto*//*Label 1121*/ 29824, // Rule ID 1673 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (DIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1673,
        GIR_Done,
      // Label 1121: @29824
      GIM_Try, /*On fail goto*//*Label 1122*/ 29847, // Rule ID 5164 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fdiv:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VDIVSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5164,
        GIR_Done,
      // Label 1122: @29847
      GIM_Reject,
    // Label 1118: @29848
    GIM_Reject,
    // Label 1109: @29849
    GIM_Try, /*On fail goto*//*Label 1123*/ 29955,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1124*/ 29885, // Rule ID 754 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fdiv:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (DIV_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 754,
        GIR_Done,
      // Label 1124: @29885
      GIM_Try, /*On fail goto*//*Label 1125*/ 29908, // Rule ID 1671 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VDIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1671,
        GIR_Done,
      // Label 1125: @29908
      GIM_Try, /*On fail goto*//*Label 1126*/ 29931, // Rule ID 1675 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (DIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1675,
        GIR_Done,
      // Label 1126: @29931
      GIM_Try, /*On fail goto*//*Label 1127*/ 29954, // Rule ID 5175 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fdiv:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VDIVSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5175,
        GIR_Done,
      // Label 1127: @29954
      GIM_Reject,
    // Label 1123: @29955
    GIM_Reject,
    // Label 1110: @29956
    GIM_Try, /*On fail goto*//*Label 1128*/ 29988, // Rule ID 755 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fdiv:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (DIV_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 755,
      GIR_Done,
    // Label 1128: @29988
    GIM_Reject,
    // Label 1111: @29989
    GIM_Try, /*On fail goto*//*Label 1129*/ 30069,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1130*/ 30022, // Rule ID 1659 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VDIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1659,
        GIR_Done,
      // Label 1130: @30022
      GIM_Try, /*On fail goto*//*Label 1131*/ 30045, // Rule ID 1667 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (DIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1667,
        GIR_Done,
      // Label 1131: @30045
      GIM_Try, /*On fail goto*//*Label 1132*/ 30068, // Rule ID 5448 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VDIVPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5448,
        GIR_Done,
      // Label 1132: @30068
      GIM_Reject,
    // Label 1129: @30069
    GIM_Reject,
    // Label 1112: @30070
    GIM_Try, /*On fail goto*//*Label 1133*/ 30150,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1134*/ 30103, // Rule ID 1657 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VDIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1657,
        GIR_Done,
      // Label 1134: @30103
      GIM_Try, /*On fail goto*//*Label 1135*/ 30126, // Rule ID 1665 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (DIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1665,
        GIR_Done,
      // Label 1135: @30126
      GIM_Try, /*On fail goto*//*Label 1136*/ 30149, // Rule ID 5430 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VDIVPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5430,
        GIR_Done,
      // Label 1136: @30149
      GIM_Reject,
    // Label 1133: @30150
    GIM_Reject,
    // Label 1113: @30151
    GIM_Try, /*On fail goto*//*Label 1137*/ 30208,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1138*/ 30184, // Rule ID 1663 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VDIVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1663,
        GIR_Done,
      // Label 1138: @30184
      GIM_Try, /*On fail goto*//*Label 1139*/ 30207, // Rule ID 5457 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VDIVPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5457,
        GIR_Done,
      // Label 1139: @30207
      GIM_Reject,
    // Label 1137: @30208
    GIM_Reject,
    // Label 1114: @30209
    GIM_Try, /*On fail goto*//*Label 1140*/ 30266,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1141*/ 30242, // Rule ID 1661 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VDIVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1661,
        GIR_Done,
      // Label 1141: @30242
      GIM_Try, /*On fail goto*//*Label 1142*/ 30265, // Rule ID 5439 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VDIVPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5439,
        GIR_Done,
      // Label 1142: @30265
      GIM_Reject,
    // Label 1140: @30266
    GIM_Reject,
    // Label 1115: @30267
    GIM_Try, /*On fail goto*//*Label 1143*/ 30298, // Rule ID 5421 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VDIVPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5421,
      GIR_Done,
    // Label 1143: @30298
    GIM_Reject,
    // Label 1116: @30299
    GIM_Try, /*On fail goto*//*Label 1144*/ 30330, // Rule ID 5412 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VDIVPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5412,
      GIR_Done,
    // Label 1144: @30330
    GIM_Reject,
    // Label 1117: @30331
    GIM_Reject,
    // Label 22: @30332
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 1148*/ 30420,
    /*GILLT_s32*//*Label 1145*/ 30341,
    /*GILLT_s64*//*Label 1146*/ 30368,
    /*GILLT_s80*//*Label 1147*/ 30395,
    // Label 1145: @30341
    GIM_Try, /*On fail goto*//*Label 1149*/ 30367, // Rule ID 822 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fneg:{ *:[f32] } RFP32:{ *:[f32] }:$src)  =>  (CHS_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp32,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 822,
      GIR_Done,
    // Label 1149: @30367
    GIM_Reject,
    // Label 1146: @30368
    GIM_Try, /*On fail goto*//*Label 1150*/ 30394, // Rule ID 823 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fneg:{ *:[f64] } RFP64:{ *:[f64] }:$src)  =>  (CHS_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp64,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 823,
      GIR_Done,
    // Label 1150: @30394
    GIM_Reject,
    // Label 1147: @30395
    GIM_Try, /*On fail goto*//*Label 1151*/ 30419, // Rule ID 824 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fneg:{ *:[f80] } RFP80:{ *:[f80] }:$src)  =>  (CHS_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 824,
      GIR_Done,
    // Label 1151: @30419
    GIM_Reject,
    // Label 1148: @30420
    GIM_Reject,
    // Label 23: @30421
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/4, 16, /*)*//*default:*//*Label 1156*/ 30696,
    /*GILLT_s64*//*Label 1152*/ 30439,
    /*GILLT_s80*//*Label 1153*/ 30575, 0, 0, 0, 0, 0,
    /*GILLT_v4s64*//*Label 1154*/ 30626, 0, 0, 0,
    /*GILLT_v8s64*//*Label 1155*/ 30672,
    // Label 1152: @30439
    GIM_Try, /*On fail goto*//*Label 1157*/ 30574,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1158*/ 30464, // Rule ID 1435 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (CVTSS2SDrr:{ *:[f64] } FR32:{ *:[f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSS2SDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1435,
        GIR_Done,
      // Label 1158: @30464
      GIM_Try, /*On fail goto*//*Label 1159*/ 30485, // Rule ID 12509 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        // (fpextend:{ *:[f64] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP32:{ *:[f32] }:$src, RFP64:{ *:[i32] })
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/42,
        // GIR_Coverage, 12509,
        GIR_Done,
      // Label 1159: @30485
      GIM_Try, /*On fail goto*//*Label 1160*/ 30529, // Rule ID 12655 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (VCVTSS2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12655,
        GIR_Done,
      // Label 1160: @30529
      GIM_Try, /*On fail goto*//*Label 1161*/ 30573, // Rule ID 14877 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        // (fpextend:{ *:[f64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTSS2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32X:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14877,
        GIR_Done,
      // Label 1161: @30573
      GIM_Reject,
    // Label 1157: @30574
    GIM_Reject,
    // Label 1153: @30575
    GIM_Try, /*On fail goto*//*Label 1162*/ 30600, // Rule ID 12510 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fpextend:{ *:[f80] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP32:{ *:[f32] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/71,
      // GIR_Coverage, 12510,
      GIR_Done,
    // Label 1162: @30600
    GIM_Try, /*On fail goto*//*Label 1163*/ 30625, // Rule ID 12511 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpextend:{ *:[f80] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP64:{ *:[f64] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/71,
      // GIR_Coverage, 12511,
      GIR_Done,
    // Label 1163: @30625
    GIM_Reject,
    // Label 1154: @30626
    GIM_Try, /*On fail goto*//*Label 1164*/ 30671,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1165*/ 30651, // Rule ID 1463 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (fpextend:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDYrr:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1463,
        GIR_Done,
      // Label 1165: @30651
      GIM_Try, /*On fail goto*//*Label 1166*/ 30670, // Rule ID 8349 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (fpextend:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8349,
        GIR_Done,
      // Label 1166: @30670
      GIM_Reject,
    // Label 1164: @30671
    GIM_Reject,
    // Label 1155: @30672
    GIM_Try, /*On fail goto*//*Label 1167*/ 30695, // Rule ID 8328 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fpextend:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTPS2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8328,
      GIR_Done,
    // Label 1167: @30695
    GIM_Reject,
    // Label 1156: @30696
    GIM_Reject,
    // Label 24: @30697
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 15, /*)*//*default:*//*Label 1172*/ 30981,
    /*GILLT_s32*//*Label 1168*/ 30715,
    /*GILLT_s64*//*Label 1169*/ 30885, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 1170*/ 30911, 0, 0, 0,
    /*GILLT_v8s32*//*Label 1171*/ 30957,
    // Label 1168: @30715
    GIM_Try, /*On fail goto*//*Label 1173*/ 30738, // Rule ID 1429 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (CVTSD2SSrr:{ *:[f32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSD2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1429,
      GIR_Done,
    // Label 1173: @30738
    GIM_Try, /*On fail goto*//*Label 1174*/ 30763, // Rule ID 12512 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpround:{ *:[f32] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP64:{ *:[f64] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/32,
      // GIR_Coverage, 12512,
      GIR_Done,
    // Label 1174: @30763
    GIM_Try, /*On fail goto*//*Label 1175*/ 30788, // Rule ID 12513 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f32] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP80:{ *:[f80] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/32,
      // GIR_Coverage, 12513,
      GIR_Done,
    // Label 1175: @30788
    GIM_Try, /*On fail goto*//*Label 1176*/ 30836, // Rule ID 12654 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (VCVTSD2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12654,
      GIR_Done,
    // Label 1176: @30836
    GIM_Try, /*On fail goto*//*Label 1177*/ 30884, // Rule ID 14881 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fpround:{ *:[f32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTSD2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64X:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14881,
      GIR_Done,
    // Label 1177: @30884
    GIM_Reject,
    // Label 1169: @30885
    GIM_Try, /*On fail goto*//*Label 1178*/ 30910, // Rule ID 12514 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f64] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP80:{ *:[f80] }:$src, RFP64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/42,
      // GIR_Coverage, 12514,
      GIR_Done,
    // Label 1178: @30910
    GIM_Reject,
    // Label 1170: @30911
    GIM_Try, /*On fail goto*//*Label 1179*/ 30956,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1180*/ 30936, // Rule ID 1475 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        // (fpround:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSYrr:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1475,
        GIR_Done,
      // Label 1180: @30936
      GIM_Try, /*On fail goto*//*Label 1181*/ 30955, // Rule ID 8319 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        // (fpround:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8319,
        GIR_Done,
      // Label 1181: @30955
      GIM_Reject,
    // Label 1179: @30956
    GIM_Reject,
    // Label 1171: @30957
    GIM_Try, /*On fail goto*//*Label 1182*/ 30980, // Rule ID 8298 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fpround:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTPD2PSZrr:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8298,
      GIR_Done,
    // Label 1182: @30980
    GIM_Reject,
    // Label 1172: @30981
    GIM_Reject,
    // Label 25: @30982
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1191*/ 31611,
    /*GILLT_s32*//*Label 1183*/ 31005,
    /*GILLT_s64*//*Label 1184*/ 31144, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1185*/ 31283, 0,
    /*GILLT_v4s32*//*Label 1186*/ 31307,
    /*GILLT_v4s64*//*Label 1187*/ 31423, 0, 0,
    /*GILLT_v8s32*//*Label 1188*/ 31470,
    /*GILLT_v8s64*//*Label 1189*/ 31540, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1190*/ 31587,
    // Label 1183: @31005
    GIM_Try, /*On fail goto*//*Label 1192*/ 31028, // Rule ID 1367 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1367,
      GIR_Done,
    // Label 1192: @31028
    GIM_Try, /*On fail goto*//*Label 1193*/ 31051, // Rule ID 1371 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1371,
      GIR_Done,
    // Label 1193: @31051
    GIM_Try, /*On fail goto*//*Label 1194*/ 31074, // Rule ID 1375 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1375,
      GIR_Done,
    // Label 1194: @31074
    GIM_Try, /*On fail goto*//*Label 1195*/ 31097, // Rule ID 1379 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1379,
      GIR_Done,
    // Label 1195: @31097
    GIM_Try, /*On fail goto*//*Label 1196*/ 31120, // Rule ID 8240 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8240,
      GIR_Done,
    // Label 1196: @31120
    GIM_Try, /*On fail goto*//*Label 1197*/ 31143, // Rule ID 8250 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8250,
      GIR_Done,
    // Label 1197: @31143
    GIM_Reject,
    // Label 1184: @31144
    GIM_Try, /*On fail goto*//*Label 1198*/ 31167, // Rule ID 1369 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1369,
      GIR_Done,
    // Label 1198: @31167
    GIM_Try, /*On fail goto*//*Label 1199*/ 31190, // Rule ID 1373 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1373,
      GIR_Done,
    // Label 1199: @31190
    GIM_Try, /*On fail goto*//*Label 1200*/ 31213, // Rule ID 1377 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1377,
      GIR_Done,
    // Label 1200: @31213
    GIM_Try, /*On fail goto*//*Label 1201*/ 31236, // Rule ID 1381 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1381,
      GIR_Done,
    // Label 1201: @31236
    GIM_Try, /*On fail goto*//*Label 1202*/ 31259, // Rule ID 8245 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8245,
      GIR_Done,
    // Label 1202: @31259
    GIM_Try, /*On fail goto*//*Label 1203*/ 31282, // Rule ID 8255 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8255,
      GIR_Done,
    // Label 1203: @31282
    GIM_Reject,
    // Label 1185: @31283
    GIM_Try, /*On fail goto*//*Label 1204*/ 31306, // Rule ID 8844 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2QQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8844,
      GIR_Done,
    // Label 1204: @31306
    GIM_Reject,
    // Label 1186: @31307
    GIM_Try, /*On fail goto*//*Label 1205*/ 31330, // Rule ID 1449 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1449,
      GIR_Done,
    // Label 1205: @31330
    GIM_Try, /*On fail goto*//*Label 1206*/ 31353, // Rule ID 1453 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (CVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1453,
      GIR_Done,
    // Label 1206: @31353
    GIM_Try, /*On fail goto*//*Label 1207*/ 31376, // Rule ID 1457 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQYrr:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1457,
      GIR_Done,
    // Label 1207: @31376
    GIM_Try, /*On fail goto*//*Label 1208*/ 31399, // Rule ID 8427 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8427,
      GIR_Done,
    // Label 1208: @31399
    GIM_Try, /*On fail goto*//*Label 1209*/ 31422, // Rule ID 8466 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8466,
      GIR_Done,
    // Label 1209: @31422
    GIM_Reject,
    // Label 1187: @31423
    GIM_Try, /*On fail goto*//*Label 1210*/ 31446, // Rule ID 8853 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2QQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8853,
      GIR_Done,
    // Label 1210: @31446
    GIM_Try, /*On fail goto*//*Label 1211*/ 31469, // Rule ID 8883 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2QQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8883,
      GIR_Done,
    // Label 1211: @31469
    GIM_Reject,
    // Label 1188: @31470
    GIM_Try, /*On fail goto*//*Label 1212*/ 31493, // Rule ID 1451 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQYrr:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1451,
      GIR_Done,
    // Label 1212: @31493
    GIM_Try, /*On fail goto*//*Label 1213*/ 31516, // Rule ID 8436 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8436,
      GIR_Done,
    // Label 1213: @31516
    GIM_Try, /*On fail goto*//*Label 1214*/ 31539, // Rule ID 8445 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2DQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8445,
      GIR_Done,
    // Label 1214: @31539
    GIM_Reject,
    // Label 1189: @31540
    GIM_Try, /*On fail goto*//*Label 1215*/ 31563, // Rule ID 8832 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2QQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8832,
      GIR_Done,
    // Label 1215: @31563
    GIM_Try, /*On fail goto*//*Label 1216*/ 31586, // Rule ID 8862 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2QQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8862,
      GIR_Done,
    // Label 1216: @31586
    GIM_Reject,
    // Label 1190: @31587
    GIM_Try, /*On fail goto*//*Label 1217*/ 31610, // Rule ID 8415 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2DQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8415,
      GIR_Done,
    // Label 1217: @31610
    GIM_Reject,
    // Label 1191: @31611
    GIM_Reject,
    // Label 26: @31612
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1226*/ 31965,
    /*GILLT_s32*//*Label 1218*/ 31635,
    /*GILLT_s64*//*Label 1219*/ 31682, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1220*/ 31729, 0,
    /*GILLT_v4s32*//*Label 1221*/ 31753,
    /*GILLT_v4s64*//*Label 1222*/ 31800, 0, 0,
    /*GILLT_v8s32*//*Label 1223*/ 31847,
    /*GILLT_v8s64*//*Label 1224*/ 31894, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1225*/ 31941,
    // Label 1218: @31635
    GIM_Try, /*On fail goto*//*Label 1227*/ 31658, // Rule ID 8260 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8260,
      GIR_Done,
    // Label 1227: @31658
    GIM_Try, /*On fail goto*//*Label 1228*/ 31681, // Rule ID 8270 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8270,
      GIR_Done,
    // Label 1228: @31681
    GIM_Reject,
    // Label 1219: @31682
    GIM_Try, /*On fail goto*//*Label 1229*/ 31705, // Rule ID 8265 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8265,
      GIR_Done,
    // Label 1229: @31705
    GIM_Try, /*On fail goto*//*Label 1230*/ 31728, // Rule ID 8275 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8275,
      GIR_Done,
    // Label 1230: @31728
    GIM_Reject,
    // Label 1220: @31729
    GIM_Try, /*On fail goto*//*Label 1231*/ 31752, // Rule ID 8904 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2UQQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8904,
      GIR_Done,
    // Label 1231: @31752
    GIM_Reject,
    // Label 1221: @31753
    GIM_Try, /*On fail goto*//*Label 1232*/ 31776, // Rule ID 8487 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UDQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8487,
      GIR_Done,
    // Label 1232: @31776
    GIM_Try, /*On fail goto*//*Label 1233*/ 31799, // Rule ID 8526 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UDQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8526,
      GIR_Done,
    // Label 1233: @31799
    GIM_Reject,
    // Label 1222: @31800
    GIM_Try, /*On fail goto*//*Label 1234*/ 31823, // Rule ID 8913 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UQQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8913,
      GIR_Done,
    // Label 1234: @31823
    GIM_Try, /*On fail goto*//*Label 1235*/ 31846, // Rule ID 8943 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UQQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8943,
      GIR_Done,
    // Label 1235: @31846
    GIM_Reject,
    // Label 1223: @31847
    GIM_Try, /*On fail goto*//*Label 1236*/ 31870, // Rule ID 8496 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UDQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8496,
      GIR_Done,
    // Label 1236: @31870
    GIM_Try, /*On fail goto*//*Label 1237*/ 31893, // Rule ID 8505 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UDQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8505,
      GIR_Done,
    // Label 1237: @31893
    GIM_Reject,
    // Label 1224: @31894
    GIM_Try, /*On fail goto*//*Label 1238*/ 31917, // Rule ID 8892 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UQQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8892,
      GIR_Done,
    // Label 1238: @31917
    GIM_Try, /*On fail goto*//*Label 1239*/ 31940, // Rule ID 8922 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UQQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8922,
      GIR_Done,
    // Label 1239: @31940
    GIM_Reject,
    // Label 1225: @31941
    GIM_Try, /*On fail goto*//*Label 1240*/ 31964, // Rule ID 8475 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2UDQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8475,
      GIR_Done,
    // Label 1240: @31964
    GIM_Reject,
    // Label 1226: @31965
    GIM_Reject,
    // Label 27: @31966
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1249*/ 32795,
    /*GILLT_s32*//*Label 1241*/ 31989,
    /*GILLT_s64*//*Label 1242*/ 32228, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1243*/ 32467, 0,
    /*GILLT_v4s32*//*Label 1244*/ 32491,
    /*GILLT_v4s64*//*Label 1245*/ 32584, 0, 0,
    /*GILLT_v8s32*//*Label 1246*/ 32654,
    /*GILLT_v8s64*//*Label 1247*/ 32724, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1248*/ 32771,
    // Label 1241: @31989
    GIM_Try, /*On fail goto*//*Label 1250*/ 32012, // Rule ID 1383 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1383,
      GIR_Done,
    // Label 1250: @32012
    GIM_Try, /*On fail goto*//*Label 1251*/ 32035, // Rule ID 1385 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SSrr:{ *:[f32] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1385,
      GIR_Done,
    // Label 1251: @32035
    GIM_Try, /*On fail goto*//*Label 1252*/ 32083, // Rule ID 12650 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12650,
      GIR_Done,
    // Label 1252: @32083
    GIM_Try, /*On fail goto*//*Label 1253*/ 32131, // Rule ID 12651 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12651,
      GIR_Done,
    // Label 1253: @32131
    GIM_Try, /*On fail goto*//*Label 1254*/ 32179, // Rule ID 14833 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14833,
      GIR_Done,
    // Label 1254: @32179
    GIM_Try, /*On fail goto*//*Label 1255*/ 32227, // Rule ID 14834 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14834,
      GIR_Done,
    // Label 1255: @32227
    GIM_Reject,
    // Label 1242: @32228
    GIM_Try, /*On fail goto*//*Label 1256*/ 32251, // Rule ID 1387 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SDrr:{ *:[f64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1387,
      GIR_Done,
    // Label 1256: @32251
    GIM_Try, /*On fail goto*//*Label 1257*/ 32274, // Rule ID 1389 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1389,
      GIR_Done,
    // Label 1257: @32274
    GIM_Try, /*On fail goto*//*Label 1258*/ 32322, // Rule ID 12652 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12652,
      GIR_Done,
    // Label 1258: @32322
    GIM_Try, /*On fail goto*//*Label 1259*/ 32370, // Rule ID 12653 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12653,
      GIR_Done,
    // Label 1259: @32370
    GIM_Try, /*On fail goto*//*Label 1260*/ 32418, // Rule ID 14835 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14835,
      GIR_Done,
    // Label 1260: @32418
    GIM_Try, /*On fail goto*//*Label 1261*/ 32466, // Rule ID 14836 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14836,
      GIR_Done,
    // Label 1261: @32466
    GIM_Reject,
    // Label 1243: @32467
    GIM_Try, /*On fail goto*//*Label 1262*/ 32490, // Rule ID 8964 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8964,
      GIR_Done,
    // Label 1262: @32490
    GIM_Reject,
    // Label 1244: @32491
    GIM_Try, /*On fail goto*//*Label 1263*/ 32514, // Rule ID 1423 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1423,
      GIR_Done,
    // Label 1263: @32514
    GIM_Try, /*On fail goto*//*Label 1264*/ 32537, // Rule ID 1427 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (CVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1427,
      GIR_Done,
    // Label 1264: @32537
    GIM_Try, /*On fail goto*//*Label 1265*/ 32560, // Rule ID 8397 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8397,
      GIR_Done,
    // Label 1265: @32560
    GIM_Try, /*On fail goto*//*Label 1266*/ 32583, // Rule ID 9033 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9033,
      GIR_Done,
    // Label 1266: @32583
    GIM_Reject,
    // Label 1245: @32584
    GIM_Try, /*On fail goto*//*Label 1267*/ 32607, // Rule ID 1470 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDYrr:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1470,
      GIR_Done,
    // Label 1267: @32607
    GIM_Try, /*On fail goto*//*Label 1268*/ 32630, // Rule ID 8376 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8376,
      GIR_Done,
    // Label 1268: @32630
    GIM_Try, /*On fail goto*//*Label 1269*/ 32653, // Rule ID 8973 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8973,
      GIR_Done,
    // Label 1269: @32653
    GIM_Reject,
    // Label 1246: @32654
    GIM_Try, /*On fail goto*//*Label 1270*/ 32677, // Rule ID 1425 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSYrr:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1425,
      GIR_Done,
    // Label 1270: @32677
    GIM_Try, /*On fail goto*//*Label 1271*/ 32700, // Rule ID 8406 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8406,
      GIR_Done,
    // Label 1271: @32700
    GIM_Try, /*On fail goto*//*Label 1272*/ 32723, // Rule ID 9012 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9012,
      GIR_Done,
    // Label 1272: @32723
    GIM_Reject,
    // Label 1247: @32724
    GIM_Try, /*On fail goto*//*Label 1273*/ 32747, // Rule ID 8358 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8358,
      GIR_Done,
    // Label 1273: @32747
    GIM_Try, /*On fail goto*//*Label 1274*/ 32770, // Rule ID 8952 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8952,
      GIR_Done,
    // Label 1274: @32770
    GIM_Reject,
    // Label 1248: @32771
    GIM_Try, /*On fail goto*//*Label 1275*/ 32794, // Rule ID 8385 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8385,
      GIR_Done,
    // Label 1275: @32794
    GIM_Reject,
    // Label 1249: @32795
    GIM_Reject,
    // Label 28: @32796
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1284*/ 33249,
    /*GILLT_s32*//*Label 1276*/ 32819,
    /*GILLT_s64*//*Label 1277*/ 32916, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1278*/ 33013, 0,
    /*GILLT_v4s32*//*Label 1279*/ 33037,
    /*GILLT_v4s64*//*Label 1280*/ 33084, 0, 0,
    /*GILLT_v8s32*//*Label 1281*/ 33131,
    /*GILLT_v8s64*//*Label 1282*/ 33178, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1283*/ 33225,
    // Label 1276: @32819
    GIM_Try, /*On fail goto*//*Label 1285*/ 32867, // Rule ID 14841 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14841,
      GIR_Done,
    // Label 1285: @32867
    GIM_Try, /*On fail goto*//*Label 1286*/ 32915, // Rule ID 14842 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14842,
      GIR_Done,
    // Label 1286: @32915
    GIM_Reject,
    // Label 1277: @32916
    GIM_Try, /*On fail goto*//*Label 1287*/ 32964, // Rule ID 14843 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14843,
      GIR_Done,
    // Label 1287: @32964
    GIM_Try, /*On fail goto*//*Label 1288*/ 33012, // Rule ID 14844 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14844,
      GIR_Done,
    // Label 1288: @33012
    GIM_Reject,
    // Label 1278: @33013
    GIM_Try, /*On fail goto*//*Label 1289*/ 33036, // Rule ID 8994 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTUQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8994,
      GIR_Done,
    // Label 1289: @33036
    GIM_Reject,
    // Label 1279: @33037
    GIM_Try, /*On fail goto*//*Label 1290*/ 33060, // Rule ID 8574 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8574,
      GIR_Done,
    // Label 1290: @33060
    GIM_Try, /*On fail goto*//*Label 1291*/ 33083, // Rule ID 9063 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9063,
      GIR_Done,
    // Label 1291: @33083
    GIM_Reject,
    // Label 1280: @33084
    GIM_Try, /*On fail goto*//*Label 1292*/ 33107, // Rule ID 8553 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8553,
      GIR_Done,
    // Label 1292: @33107
    GIM_Try, /*On fail goto*//*Label 1293*/ 33130, // Rule ID 9003 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9003,
      GIR_Done,
    // Label 1293: @33130
    GIM_Reject,
    // Label 1281: @33131
    GIM_Try, /*On fail goto*//*Label 1294*/ 33154, // Rule ID 8583 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8583,
      GIR_Done,
    // Label 1294: @33154
    GIM_Try, /*On fail goto*//*Label 1295*/ 33177, // Rule ID 9042 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9042,
      GIR_Done,
    // Label 1295: @33177
    GIM_Reject,
    // Label 1282: @33178
    GIM_Try, /*On fail goto*//*Label 1296*/ 33201, // Rule ID 8535 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8535,
      GIR_Done,
    // Label 1296: @33201
    GIM_Try, /*On fail goto*//*Label 1297*/ 33224, // Rule ID 8982 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8982,
      GIR_Done,
    // Label 1297: @33224
    GIM_Reject,
    // Label 1283: @33225
    GIM_Try, /*On fail goto*//*Label 1298*/ 33248, // Rule ID 8562 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTUDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8562,
      GIR_Done,
    // Label 1298: @33248
    GIM_Reject,
    // Label 1284: @33249
    GIM_Reject,
    // Label 29: @33250
    GIM_Try, /*On fail goto*//*Label 1299*/ 33262, // Rule ID 552 //
      // MIs[0] dst
      GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
      // (br (bb:{ *:[Other] }):$dst)  =>  (JMP_1 (bb:{ *:[Other] }):$dst)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::JMP_1,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 552,
      GIR_Done,
    // Label 1299: @33262
    GIM_Reject,
    // Label 30: @33263
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 5, /*)*//*default:*//*Label 1303*/ 33352,
    /*GILLT_s16*//*Label 1300*/ 33272,
    /*GILLT_s32*//*Label 1301*/ 33308,
    /*GILLT_s64*//*Label 1302*/ 33330,
    // Label 1300: @33272
    GIM_Try, /*On fail goto*//*Label 1304*/ 33307, // Rule ID 15891 //
      GIM_CheckFeatures, GIFBS_HasMOVBE,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (bswap:{ *:[i16] } GR16:{ *:[i16] }:$src)  =>  (ROL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src, 8:{ *:[i8] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ROL16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_AddImm, /*InsnID*/0, /*Imm*/8,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15891,
      GIR_Done,
    // Label 1304: @33307
    GIM_Reject,
    // Label 1301: @33308
    GIM_Try, /*On fail goto*//*Label 1305*/ 33329, // Rule ID 5 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bswap:{ *:[i32] } GR32:{ *:[i32] }:$src)  =>  (BSWAP32r:{ *:[i32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP32r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5,
      GIR_Done,
    // Label 1305: @33329
    GIM_Reject,
    // Label 1302: @33330
    GIM_Try, /*On fail goto*//*Label 1306*/ 33351, // Rule ID 6 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bswap:{ *:[i64] } GR64:{ *:[i64] }:$src)  =>  (BSWAP64r:{ *:[i64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP64r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6,
      GIR_Done,
    // Label 1306: @33351
    GIM_Reject,
    // Label 1303: @33352
    GIM_Reject,
    // Label 31: @33353
    GIM_Reject,
    };
  return MatchTable0;
}
#endif // ifdef GET_GLOBALISEL_IMPL
#ifdef GET_GLOBALISEL_PREDICATES_DECL
PredicateBitset AvailableModuleFeatures;
mutable PredicateBitset AvailableFunctionFeatures;
PredicateBitset getAvailableFeatures() const {
  return AvailableModuleFeatures | AvailableFunctionFeatures;
}
PredicateBitset
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const;
PredicateBitset
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget,
                                 const MachineFunction *MF) const;
#endif // ifdef GET_GLOBALISEL_PREDICATES_DECL
#ifdef GET_GLOBALISEL_PREDICATES_INIT
AvailableModuleFeatures(computeAvailableModuleFeatures(&STI)),
AvailableFunctionFeatures()
#endif // ifdef GET_GLOBALISEL_PREDICATES_INIT
