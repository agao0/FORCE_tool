// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

namespace sym {

/**
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: BodyJacobian
 *
 * Args:
 *     joint_angles: Matrix71
 *
 * Outputs:
 *     res: Matrix67
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 6, 7> Leftlowersensorbodyjacobian(
    const Eigen::Matrix<Scalar, 7, 1>& joint_angles) {
  // Total ops: 438

  // Input arrays

  // Intermediate terms (114)
  const Scalar _tmp0 = std::cos(joint_angles(4, 0));
  const Scalar _tmp1 = std::cos(joint_angles(2, 0));
  const Scalar _tmp2 = 1 - std::cos(joint_angles(0, 0));
  const Scalar _tmp3 = Scalar(0.36863892073799998) * _tmp2;
  const Scalar _tmp4 = std::sin(joint_angles(0, 0));
  const Scalar _tmp5 = Scalar(0.47771400000000003) * _tmp4;
  const Scalar _tmp6 = _tmp3 + _tmp5;
  const Scalar _tmp7 = 1 - std::cos(joint_angles(1, 0));
  const Scalar _tmp8 = 1 - Scalar(0.64784630232100004) * _tmp7;
  const Scalar _tmp9 = 1 - Scalar(0.728210975245) * _tmp2;
  const Scalar _tmp10 = Scalar(0.47764205971399998) * _tmp7;
  const Scalar _tmp11 = Scalar(0.249048550476) * _tmp2;
  const Scalar _tmp12 = Scalar(0.70710700000000004) * _tmp4;
  const Scalar _tmp13 = _tmp11 - _tmp12;
  const Scalar _tmp14 = std::sin(joint_angles(1, 0));
  const Scalar _tmp15 = Scalar(0.80488899999999997) * _tmp14;
  const Scalar _tmp16 = -_tmp10 * _tmp9 + _tmp13 * _tmp15 + _tmp6 * _tmp8;
  const Scalar _tmp17 = 1 - Scalar(0.35215441747600001) * _tmp7;
  const Scalar _tmp18 = Scalar(0.59342600000000001) * _tmp14;
  const Scalar _tmp19 = -_tmp10 * _tmp6 + _tmp13 * _tmp18 + _tmp17 * _tmp9;
  const Scalar _tmp20 = std::sin(joint_angles(2, 0));
  const Scalar _tmp21 = _tmp1 * _tmp16 - _tmp19 * _tmp20;
  const Scalar _tmp22 = _tmp1 * _tmp19 + _tmp16 * _tmp20;
  const Scalar _tmp23 = std::cos(joint_angles(3, 0));
  const Scalar _tmp24 = 1 - Scalar(1.000000719797) * _tmp7;
  const Scalar _tmp25 = _tmp13 * _tmp24 - _tmp15 * _tmp6 - _tmp18 * _tmp9;
  const Scalar _tmp26 = std::sin(joint_angles(3, 0));
  const Scalar _tmp27 = _tmp22 * _tmp23 + _tmp25 * _tmp26;
  const Scalar _tmp28 = std::sin(joint_angles(4, 0));
  const Scalar _tmp29 = _tmp0 * _tmp21 - _tmp27 * _tmp28;
  const Scalar _tmp30 = _tmp22 * _tmp26;
  const Scalar _tmp31 = _tmp23 * _tmp25 - _tmp30;
  const Scalar _tmp32 = _tmp0 * _tmp27 + _tmp21 * _tmp28;
  const Scalar _tmp33 = -_tmp23;
  const Scalar _tmp34 = _tmp33 + 1;
  const Scalar _tmp35 = Scalar(0.30359999999999998) * _tmp34;
  const Scalar _tmp36 = -_tmp25 * _tmp35 + Scalar(0.0053400000000000001) * _tmp29 -
                        Scalar(0.30359999999999998) * _tmp30 -
                        Scalar(0.51190000000000002) * _tmp31 -
                        Scalar(0.029790000000000001) * _tmp32;
  const Scalar _tmp37 = Scalar(0.337794913398) * _tmp2;
  const Scalar _tmp38 = Scalar(0.52133399999999996) * _tmp4;
  const Scalar _tmp39 = _tmp37 - _tmp38;
  const Scalar _tmp40 = 1 - Scalar(0.77178944900500002) * _tmp2;
  const Scalar _tmp41 = _tmp11 + _tmp12;
  const Scalar _tmp42 = -_tmp10 * _tmp41 + _tmp15 * _tmp40 + _tmp39 * _tmp8;
  const Scalar _tmp43 = -_tmp10 * _tmp39 + _tmp17 * _tmp41 + _tmp18 * _tmp40;
  const Scalar _tmp44 = _tmp1 * _tmp43 + _tmp20 * _tmp42;
  const Scalar _tmp45 = -_tmp15 * _tmp39 - _tmp18 * _tmp41 + _tmp24 * _tmp40;
  const Scalar _tmp46 = _tmp23 * _tmp44 + _tmp26 * _tmp45;
  const Scalar _tmp47 = _tmp1 * _tmp42 - _tmp20 * _tmp43;
  const Scalar _tmp48 = _tmp0 * _tmp47 - _tmp28 * _tmp46;
  const Scalar _tmp49 = _tmp0 * _tmp46 + _tmp28 * _tmp47;
  const Scalar _tmp50 = _tmp23 * _tmp45 - _tmp26 * _tmp44;
  const Scalar _tmp51 = Scalar(0.30359999999999998) * _tmp26;
  const Scalar _tmp52 =
      -_tmp35 * _tmp45 - _tmp44 * _tmp51 + Scalar(0.0053400000000000001) * _tmp48 -
      Scalar(0.029790000000000001) * _tmp49 - Scalar(0.51190000000000002) * _tmp50;
  const Scalar _tmp53 = 1 - Scalar(0.499999805352) * _tmp2;
  const Scalar _tmp54 = _tmp3 - _tmp5;
  const Scalar _tmp55 = _tmp37 + _tmp38;
  const Scalar _tmp56 = -_tmp10 * _tmp53 + _tmp17 * _tmp54 + _tmp18 * _tmp55;
  const Scalar _tmp57 = -_tmp10 * _tmp54 + _tmp15 * _tmp55 + _tmp53 * _tmp8;
  const Scalar _tmp58 = _tmp1 * _tmp56 + _tmp20 * _tmp57;
  const Scalar _tmp59 = -_tmp15 * _tmp53 - _tmp18 * _tmp54 + _tmp24 * _tmp55;
  const Scalar _tmp60 = _tmp23 * _tmp58 + _tmp26 * _tmp59;
  const Scalar _tmp61 = _tmp1 * _tmp57 - _tmp20 * _tmp56;
  const Scalar _tmp62 = _tmp0 * _tmp61 - _tmp28 * _tmp60;
  const Scalar _tmp63 = _tmp0 * _tmp60 + _tmp28 * _tmp61;
  const Scalar _tmp64 = _tmp23 * _tmp59 - _tmp26 * _tmp58;
  const Scalar _tmp65 =
      -_tmp35 * _tmp59 - _tmp51 * _tmp58 + Scalar(0.0053400000000000001) * _tmp62 -
      Scalar(0.029790000000000001) * _tmp63 - Scalar(0.51190000000000002) * _tmp64;
  const Scalar _tmp66 = _tmp1 * _tmp18 + _tmp15 * _tmp20;
  const Scalar _tmp67 = _tmp26 * _tmp66;
  const Scalar _tmp68 = _tmp23 * _tmp24 - _tmp67;
  const Scalar _tmp69 = _tmp1 * _tmp15 - _tmp18 * _tmp20;
  const Scalar _tmp70 = _tmp23 * _tmp66 + _tmp24 * _tmp26;
  const Scalar _tmp71 = _tmp0 * _tmp70 + _tmp28 * _tmp69;
  const Scalar _tmp72 = _tmp0 * _tmp69 - _tmp28 * _tmp70;
  const Scalar _tmp73 = -_tmp24 * _tmp35 - Scalar(0.30359999999999998) * _tmp67 -
                        Scalar(0.51190000000000002) * _tmp68 -
                        Scalar(0.029790000000000001) * _tmp71 +
                        Scalar(0.0053400000000000001) * _tmp72;
  const Scalar _tmp74 = -_tmp1 * _tmp10;
  const Scalar _tmp75 = -_tmp17 * _tmp20 + _tmp74;
  const Scalar _tmp76 = _tmp10 * _tmp20;
  const Scalar _tmp77 = _tmp1 * _tmp17 - _tmp76;
  const Scalar _tmp78 = -_tmp18 * _tmp26 + _tmp23 * _tmp77;
  const Scalar _tmp79 = _tmp0 * _tmp75 - _tmp28 * _tmp78;
  const Scalar _tmp80 = _tmp26 * _tmp77;
  const Scalar _tmp81 = -_tmp18 * _tmp23 - _tmp80;
  const Scalar _tmp82 = _tmp0 * _tmp78 + _tmp28 * _tmp75;
  const Scalar _tmp83 = _tmp14 * _tmp34;
  const Scalar _tmp84 = Scalar(0.0053400000000000001) * _tmp79 -
                        Scalar(0.30359999999999998) * _tmp80 -
                        Scalar(0.51190000000000002) * _tmp81 -
                        Scalar(0.029790000000000001) * _tmp82 + Scalar(0.1801641336) * _tmp83;
  const Scalar _tmp85 = _tmp1 * _tmp8 + _tmp76;
  const Scalar _tmp86 = _tmp20 * _tmp8 + _tmp74;
  const Scalar _tmp87 = -_tmp15 * _tmp26 + _tmp23 * _tmp86;
  const Scalar _tmp88 = _tmp0 * _tmp85 - _tmp28 * _tmp87;
  const Scalar _tmp89 = _tmp26 * _tmp86;
  const Scalar _tmp90 = _tmp0 * _tmp87 + _tmp28 * _tmp85;
  const Scalar _tmp91 = -_tmp15 * _tmp23 - _tmp89;
  const Scalar _tmp92 =
      Scalar(0.24436430040000001) * _tmp83 + Scalar(0.0053400000000000001) * _tmp88 -
      Scalar(0.30359999999999998) * _tmp89 - Scalar(0.029790000000000001) * _tmp90 -
      Scalar(0.51190000000000002) * _tmp91;
  const Scalar _tmp93 = _tmp20 * _tmp28;
  const Scalar _tmp94 = _tmp0 * _tmp1;
  const Scalar _tmp95 = _tmp23 * _tmp94 - _tmp93;
  const Scalar _tmp96 = Scalar(0.20830000000000001) * _tmp26;
  const Scalar _tmp97 = _tmp0 * _tmp20;
  const Scalar _tmp98 = _tmp1 * _tmp28;
  const Scalar _tmp99 = -_tmp23 * _tmp98 - _tmp97;
  const Scalar _tmp100 = _tmp1 * _tmp96 - Scalar(0.029790000000000001) * _tmp95 +
                         Scalar(0.0053400000000000001) * _tmp99;
  const Scalar _tmp101 = -_tmp23 * _tmp93 + _tmp94;
  const Scalar _tmp102 = _tmp23 * _tmp97 + _tmp98;
  const Scalar _tmp103 = Scalar(0.0053400000000000001) * _tmp101 -
                         Scalar(0.029790000000000001) * _tmp102 + _tmp20 * _tmp96;
  const Scalar _tmp104 = _tmp26 * _tmp28;
  const Scalar _tmp105 = _tmp0 * _tmp26;
  const Scalar _tmp106 = Scalar(0.029790000000000001) * _tmp0;
  const Scalar _tmp107 = -Scalar(0.0053400000000000001) * _tmp104 - _tmp106 * _tmp26 -
                         Scalar(0.51190000000000002) * _tmp23 - _tmp35;
  const Scalar _tmp108 = _tmp107 * _tmp23;
  const Scalar _tmp109 = Scalar(0.30359999999999998) * _tmp23;
  const Scalar _tmp110 = Scalar(0.0053400000000000001) * _tmp28;
  const Scalar _tmp111 = -_tmp106 * _tmp23 - _tmp110 * _tmp23 + _tmp96;
  const Scalar _tmp112 =
      Scalar(0.0053400000000000001) * _tmp0 - Scalar(0.029790000000000001) * _tmp28;
  const Scalar _tmp113 = -_tmp106 - _tmp110;

  // Output terms (1)
  Eigen::Matrix<Scalar, 6, 7> _res;

  _res(0, 0) = -Scalar(0.70710700000000004) * _tmp29 * _tmp52 +
               Scalar(0.47771400000000003) * _tmp29 * _tmp65 +
               Scalar(0.70710700000000004) * _tmp36 * _tmp48 -
               Scalar(0.47771400000000003) * _tmp36 * _tmp62 -
               Scalar(0.52133399999999996) * _tmp48 * _tmp65 +
               Scalar(0.52133399999999996) * _tmp52 * _tmp62;
  _res(1, 0) = -Scalar(0.70710700000000004) * _tmp32 * _tmp52 +
               Scalar(0.47771400000000003) * _tmp32 * _tmp65 +
               Scalar(0.70710700000000004) * _tmp36 * _tmp49 -
               Scalar(0.47771400000000003) * _tmp36 * _tmp63 -
               Scalar(0.52133399999999996) * _tmp49 * _tmp65 +
               Scalar(0.52133399999999996) * _tmp52 * _tmp63;
  _res(2, 0) = -Scalar(0.70710700000000004) * _tmp31 * _tmp52 +
               Scalar(0.47771400000000003) * _tmp31 * _tmp65 +
               Scalar(0.70710700000000004) * _tmp36 * _tmp50 -
               Scalar(0.47771400000000003) * _tmp36 * _tmp64 -
               Scalar(0.52133399999999996) * _tmp50 * _tmp65 +
               Scalar(0.52133399999999996) * _tmp52 * _tmp64;
  _res(3, 0) = Scalar(0.52133399999999996) * _tmp29 + Scalar(0.47771400000000003) * _tmp48 +
               Scalar(0.70710700000000004) * _tmp62;
  _res(4, 0) = Scalar(0.52133399999999996) * _tmp32 + Scalar(0.47771400000000003) * _tmp49 +
               Scalar(0.70710700000000004) * _tmp63;
  _res(5, 0) = Scalar(0.52133399999999996) * _tmp31 + Scalar(0.47771400000000003) * _tmp50 +
               Scalar(0.70710700000000004) * _tmp64;
  _res(0, 1) = Scalar(0.59342600000000001) * _tmp72 * _tmp84 +
               Scalar(0.80488899999999997) * _tmp72 * _tmp92 -
               Scalar(0.59342600000000001) * _tmp73 * _tmp79 -
               Scalar(0.80488899999999997) * _tmp73 * _tmp88;
  _res(1, 1) = Scalar(0.59342600000000001) * _tmp71 * _tmp84 +
               Scalar(0.80488899999999997) * _tmp71 * _tmp92 -
               Scalar(0.59342600000000001) * _tmp73 * _tmp82 -
               Scalar(0.80488899999999997) * _tmp73 * _tmp90;
  _res(2, 1) = Scalar(0.59342600000000001) * _tmp68 * _tmp84 +
               Scalar(0.80488899999999997) * _tmp68 * _tmp92 -
               Scalar(0.59342600000000001) * _tmp73 * _tmp81 -
               Scalar(0.80488899999999997) * _tmp73 * _tmp91;
  _res(3, 1) = -Scalar(0.80488899999999997) * _tmp79 + Scalar(0.59342600000000001) * _tmp88;
  _res(4, 1) = -Scalar(0.80488899999999997) * _tmp82 + Scalar(0.59342600000000001) * _tmp90;
  _res(5, 1) = -Scalar(0.80488899999999997) * _tmp81 + Scalar(0.59342600000000001) * _tmp91;
  _res(0, 2) = _tmp100 * _tmp101 - _tmp103 * _tmp99;
  _res(1, 2) = _tmp100 * _tmp102 - _tmp103 * _tmp95;
  _res(2, 2) = _tmp1 * _tmp103 * _tmp26 - _tmp100 * _tmp20 * _tmp26;
  _res(3, 2) = _tmp104;
  _res(4, 2) = -_tmp105;
  _res(5, 2) = _tmp33;
  _res(0, 3) = -_tmp104 * _tmp111 + _tmp108 * _tmp28 + _tmp109 * _tmp28;
  _res(1, 3) = -_tmp0 * _tmp108 - _tmp0 * _tmp109 + _tmp105 * _tmp111;
  _res(2, 3) = _tmp107 * _tmp26 + _tmp111 * _tmp23 + _tmp51;
  _res(3, 3) = _tmp0;
  _res(4, 3) = _tmp28;
  _res(5, 3) = 0;
  _res(0, 4) = _tmp0 * _tmp113 + _tmp112 * _tmp28;
  _res(1, 4) = -_tmp0 * _tmp112 + _tmp113 * _tmp28;
  _res(2, 4) = 0;
  _res(3, 4) = 0;
  _res(4, 4) = 0;
  _res(5, 4) = -1;
  _res(0, 5) = 0;
  _res(1, 5) = 0;
  _res(2, 5) = 0;
  _res(3, 5) = 0;
  _res(4, 5) = 0;
  _res(5, 5) = 0;
  _res(0, 6) = 0;
  _res(1, 6) = 0;
  _res(2, 6) = 0;
  _res(3, 6) = 0;
  _res(4, 6) = 0;
  _res(5, 6) = 0;

  return _res;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
