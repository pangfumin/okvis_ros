#ifndef  _CONVERTOR_H_
#define  _CONVERTOR_H_

#include <okvis/kinematics/Transformation.hpp>
#include <sophus/se3.h>
namespace svo {
    okvis::kinematics::Transformation SE32Transformation(Sophus::SE3& se3) {
        return  okvis::kinematics::Transformation(se3.matrix());
    }
    Sophus::SE3  Transformation2SE3(okvis::kinematics::Transformation& transformation) {
        Eigen::Quaterniond q(transformation.C());
        return  Sophus::SE3(q, transformation.r());
    }
}

#endif