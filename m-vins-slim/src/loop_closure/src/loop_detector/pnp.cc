#include "loop_detector/pnp.h"

#include "data_common/constants.h"
#include "math_common/math.h"

constexpr size_t kPnPInlierNumThreshold = 20u;
const int kBaseCamIdx = 0;

namespace loop_closure {

std::vector<double> SolveQuartic(const Eigen::MatrixXd& p) {
    const double A = p(0, 0);
    const double B = p(1, 0);
    const double C = p(2, 0);
    const double D = p(3, 0);
    const double E = p(4, 0);

    const double A_pw2 = A * A;
    const double B_pw2 = B * B;
    const double A_pw3 = A_pw2 * A;
    const double B_pw3 = B_pw2 * B;
    const double A_pw4 = A_pw3 * A;
    const double B_pw4 = B_pw3 * B;

    const double alpha = -3. * B_pw2 / (8. * A_pw2) + C / A;
    const double beta = B_pw3 / (8. * A_pw3) - B * C / (2. * A_pw2) + D / A;
    const double gamma =
        -3. * B_pw4 / (256. * A_pw4) + B_pw2 * C / (16. * A_pw3) -
        B * D / (4. * A_pw2) + E / A;

    const double alpha_pw2 = alpha * alpha;
    const double alpha_pw3 = alpha_pw2 * alpha;

    const std::complex<double> P(-alpha_pw2 / 12. - gamma, 0.);
    const std::complex<double> Q(
        -alpha_pw3 / 108. + alpha * gamma / 3. - std::pow(beta, 2.) / 8., 0.);
    const std::complex<double> R = -Q / 2.0 + std::sqrt(
        std::pow(Q, 2.0) / 4.0 + std::pow(P, 3.0) / 27.0);

    const std::complex<double> U = std::pow(R, (1.0 / 3.0));
    std::complex<double> y;

    if (U.real() == 0.) {
        y = -5.0 * alpha / 6.0 - std::pow(Q, (1.0 / 3.0));
    } else {
        y = -5.0 * alpha / 6.0 - P / (3.0 * U) + U;
    }

    const std::complex<double> w = std::sqrt(alpha + 2.0 * y);

    std::vector<double> real_roots;
    std::complex<double> temp;
    temp = -B / (4.0 * A) +
           0.5 * (w + std::sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
    real_roots.push_back(temp.real());
    temp = -B / (4.0 * A) +
           0.5 * (w - std::sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
    real_roots.push_back(temp.real());
    temp = -B / (4.0 * A) + 0.5 * (-w + std::sqrt(
        -(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
    real_roots.push_back(temp.real());
    temp = -B / (4.0 * A) + 0.5 * (-w - std::sqrt(
        -(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
    real_roots.push_back(temp.real());

    return real_roots;
}

Eigen::Matrix3d arun(const Eigen::MatrixXd& Hcross) {
    // Decompose matrix H to obtain rotation.
    const Eigen::JacobiSVD<Eigen::MatrixXd> SVD_cross(
        Hcross,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3d V = SVD_cross.matrixV();
    Eigen::Matrix3d U = SVD_cross.matrixU();
    Eigen::Matrix3d R = V * U.transpose();

    // Modify the result in case the rotation has determinant = -1.
    if (R.determinant() < 0) {
        Eigen::Matrix3d V_prime;
        V_prime.col(0) = V.col(0);
        V_prime.col(1) = V.col(1);
        V_prime.col(2) = -V.col(2);
        R = V_prime * U.transpose();
    }

    return R;
}

common::EigenMatrix34d arun_complete(
    const common::EigenVector3dVec& p1,
    const common::EigenVector3dVec& p2) {
    CHECK_EQ(p1.size(), p2.size());

    // Derive the centroid of the two point-clouds.
    Eigen::Vector3d points_center_1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d points_center_2 = Eigen::Vector3d::Zero();

    for (size_t i = 0u; i < p1.size(); ++i) {
        points_center_1 += p1[i];
        points_center_2 += p2[i];
    }

    points_center_1 = points_center_1 / p1.size();
    points_center_2 = points_center_2 / p2.size();

    // Compute the matrix H = sum(f'*f^{T}).
    Eigen::Matrix3d Hcross = Eigen::Matrix3d::Zero();

    for (size_t i = 0u; i < p1.size(); ++i) {
        Eigen::Vector3d f = p1[i] - points_center_1;
        Eigen::Vector3d fprime = p2[i] - points_center_2;
        Hcross += fprime * f.transpose();
    }

    // Decompose this matrix (SVD) to obtain rotation.
    const Eigen::Matrix3d rotation = arun(Hcross);
    const Eigen::Vector3d translation =
        points_center_1 - rotation * points_center_2;
    common::EigenMatrix34d solution;
    solution.block<3, 3>(0, 0) = rotation;
    solution.col(3) = translation;

    return solution;
}

void P3PCore(
    const Eigen::Vector3d& global_point_0,
    const Eigen::Vector3d& global_point_1,
    const Eigen::Vector3d& global_point_2,
    const Eigen::Vector3d& f1,
    const Eigen::Vector3d& f2,
    const Eigen::Vector3d& f3,
    common::EigenMatrix34dVec* solutions) {
    common::EigenVector3dVec points;
    points.push_back(global_point_0);
    points.push_back(global_point_1);
    points.push_back(global_point_2);

    const Eigen::Vector3d A = global_point_0;
    const Eigen::Vector3d B = global_point_1;
    const Eigen::Vector3d C = global_point_2;

    const double AB = (A - B).norm();
    const double BC = (B - C).norm();
    const double AC = (A - C).norm();

    const double cos_alpha = f2.transpose() * f3;
    const double cosbeta = f1.transpose() * f3;
    const double cosgamma = f1.transpose() * f2;

    const double a = std::pow((BC / AB), 2.);
    const double b = std::pow((AC / AB), 2.);
    const double p = 2. * cos_alpha;
    const double q = 2. * cosbeta;
    const double r = 2. * cosgamma;

    const double aSq = a * a;
    const double bSq = b * b;
    const double pSq = p * p;
    const double qSq = q * q;
    const double rSq = r * r;

    if ((pSq + qSq + rSq - p * q * r - 1.) == 0.) {
        return;
    }

    Eigen::Matrix<double, 5, 1> factors;

    factors[0] = -2. * b + bSq + aSq + 1. - b * rSq * a + 2. * b * a - 2. * a;

    if (factors[0] == 0.)
        return;

    factors[1] = -2. * b * q * a - 2. * aSq * q + b * rSq * q * a - 2. * q
        + 2. * b * q + 4. * a * q + p * b * r + b * r * p * a - bSq * r * p;
    factors[2] = qSq + bSq * rSq - b * pSq - q * p * b * r + bSq * pSq
        - b * rSq * a + 2. - 2. * bSq - a * b * r * p * q + 2. * aSq - 4. * a
        - 2. * qSq * a + qSq * aSq;
    factors[3] = -bSq * r * p + b * r * p * a - 2. * aSq * q + q * pSq * b
        + 2. * b * q * a + 4. * a * q + p * b * r - 2. * b * q - 2. * q;
    factors[4] = 1. - 2. * a + 2. * b + bSq - b * pSq + aSq - 2. * b * a;

    const std::vector<double> x_temp = SolveQuartic(factors);
    Eigen::Matrix<double, 4, 1> x;
    for (size_t i = 0u; i < 4u; ++i) {
        x[i] = x_temp[i];
    }

    const double temp =
        (pSq * (a - 1. + b) + p * q * r - q * a * r * p + (a - 1. - b) * rSq);
    const double b0 = b * temp * temp;

    const double rCb = rSq * r;

    Eigen::Matrix<double, 4, 1> tempXP2;
    tempXP2[0] = x[0] * x[0];
    tempXP2[1] = x[1] * x[1];
    tempXP2[2] = x[2] * x[2];
    tempXP2[3] = x[3] * x[3];
    Eigen::Matrix<double, 4, 1> tempXP3;
    tempXP3[0] = tempXP2[0] * x[0];
    tempXP3[1] = tempXP2[1] * x[1];
    tempXP3[2] = tempXP2[2] * x[2];
    tempXP3[3] = tempXP2[3] * x[3];

    Eigen::Matrix<double, 4, 1> ones;
    for (size_t i = 0u; i < 4u; ++i) {
        ones[i] = 1.0;
    }

    const Eigen::Matrix<double, 4, 1> b1_part1 =
            (1. - a - b) * tempXP2 + (q * a - q) * x + (1. - a + b) * ones;

    const Eigen::Matrix<double, 4, 1> b1_part2 =
        (aSq * rCb + 2. * b * rCb * a - b * rSq * rCb * a - 2. * a * rCb
            + rCb + bSq * rCb - 2. * rCb * b) * tempXP3
        + (p * rSq + p * aSq * rSq - 2. * b * rCb * q * a + 2. * rCb * b * q
            - 2. * rCb * q - 2. * p * (a + b) * rSq + rSq * rSq * p * b
            + 4. * a * rCb * q + b * q * a * rCb * rSq - 2. * rCb * aSq * q
            + 2. * rSq * p * b * a + bSq * rSq * p - rSq * rSq * p * bSq)
            * tempXP2
        + (rCb * qSq + rSq * rCb * bSq + r * pSq * bSq - 4. * a * rCb
            - 2. * a * rCb * qSq + rCb * qSq * aSq + 2. * aSq * rCb
            - 2. * bSq * rCb - 2. * pSq * b * r + 4. * p * a * rSq * q
            + 2. * a * pSq * r * b - 2. * a * rSq * q * b * p
            - 2. * pSq * a * r + r * pSq - b * rSq * rCb * a
            + 2. * p * rSq * b * q + r * pSq * aSq - 2. * p * q * rSq
            + 2. * rCb - 2. * rSq * p * aSq * q - rSq * rSq * q * b * p) * x
        + (4. * a * rCb * q + p * rSq * qSq + 2. * pSq * p * b * a
            - 4. * p * a * rSq - 2. * rCb * b * q - 2. * pSq * q * r
            - 2. * bSq * rSq * p + rSq * rSq * p * b + 2. * p * aSq * rSq
            - 2. * rCb * aSq * q - 2. * pSq * p * a + pSq * p * aSq
            + 2. * p * rSq + pSq * p + 2. * b * rCb * q * a
            + 2. * q * pSq * b * r + 4. * q * a * r * pSq
            - 2. * p * a * rSq * qSq - 2. * pSq * aSq * r * q
            + p * aSq * rSq * qSq - 2. * rCb * q - 2. * pSq * p * b
            + pSq * p * bSq - 2. * pSq * b * r * q * a) * ones;

    Eigen::Matrix<double, 4, 1> b1;
    b1[0] = b1_part1[0] * b1_part2[0];
    b1[1] = b1_part1[1] * b1_part2[1];
    b1[2] = b1_part1[2] * b1_part2[2];
    b1[3] = b1_part1[3] * b1_part2[3];

    const Eigen::Matrix<double, 4, 1> y = b1 / b0;
    Eigen::Matrix<double, 4 , 1> tempYP2;
    tempYP2[0] = std::pow(y[0], 2);
    tempYP2[1] = std::pow(y[1], 2);
    tempYP2[2] = std::pow(y[2], 2);
    tempYP2[3] = std::pow(y[3], 2);

    Eigen::Matrix<double, 4, 1> tempXY;
    tempXY[0] = x[0] * y[0];
    tempXY[1] = x[1] * y[1];
    tempXY[2] = x[2] * y[2];
    tempXY[3] = x[3] * y[3];

    const Eigen::Matrix<double, 4, 1> v = tempXP2 + tempYP2 - r * tempXY;

    Eigen::Matrix<double, 4, 1> Z;
    Z[0] = AB / std::sqrt(v[0]);
    Z[1] = AB / std::sqrt(v[1]);
    Z[2] = AB / std::sqrt(v[2]);
    Z[3] = AB / std::sqrt(v[3]);

    Eigen::Matrix<double, 4, 1> X;
    X[0] = x[0] * Z[0];
    X[1] = x[1] * Z[1];
    X[2] = x[2] * Z[2];
    X[3] = x[3] * Z[3];

    Eigen::Matrix<double, 4, 1> Y;
    Y[0] = y[0] * Z[0];
    Y[1] = y[1] * Z[1];
    Y[2] = y[2] * Z[2];
    Y[3] = y[3] * Z[3];

    for(int i = 0; i < 4; ++i) {
        // Apply arun to find the transformation.
        common::EigenVector3dVec p_cam;
        p_cam.push_back(X[i] * f1);
        p_cam.push_back(Y[i] * f2);
        p_cam.push_back(Z[i] * f3);

        // cam-->point.
        common::EigenMatrix34d solution = arun_complete(points, p_cam);
        solutions->push_back(solution);
    }
}

bool OptimizePnPGaussNewton(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const std::vector<aslam::Transformation>& T_C0toCi,
    const Eigen::Matrix3Xd& p_LinMs,
    const std::vector<int>& inliers_init,
    const aslam::NCamera::ConstPtr& cameras,
    const double converge_tolerance,
    const int max_num_iterations,
    const std::vector<aslam::Transformation>& T_OtoCi,
    const aslam::Transformation& T_OtoM_init,
    aslam::Transformation* T_OtoM_final_ptr,
    Eigen::Matrix<double, 6, 6>* H_ptr,
    double* cost_init_ptr,
    double* cost_final_ptr,
    int* num_iterations_ptr) {
    CHECK_NOTNULL(T_OtoM_final_ptr);
    CHECK_NOTNULL(cost_init_ptr);
    CHECK_NOTNULL(cost_final_ptr);
    CHECK_NOTNULL(num_iterations_ptr);
    CHECK_EQ(keypoints.cols(), p_LinMs.cols());
    if (inliers_init.size() < 6) {
        return false;
    }

    *cost_init_ptr = -1;
    *num_iterations_ptr = 0;
    bool is_converged = false;
    double cost_curr;

    aslam::Transformation T_MtoO_optimized = T_OtoM_init.inverse();
    aslam::Transformation T_MtoC0_optimized =
        T_OtoCi[kBaseCamIdx] * T_MtoO_optimized;

    while (true) {
        // Calculate H and g for H * epsilon = g.
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

        cost_curr = 0;
        for (int inlier : inliers_init) {
            const Eigen::Vector2d meas = keypoints.col(inlier);
            const Eigen::Vector3d p_LinM = p_LinMs.col(inlier);
            const int cam_idx = cam_indices(inlier);
            Eigen::Vector3d p_LinC0 = (T_C0toCi[cam_idx] * T_MtoC0_optimized).transform(p_LinM);

            CHECK_GT(std::fabs(p_LinC0(2)), common::kEpsilon);
            Eigen::Matrix<double, 2, 3> jacobian_pi =
                Eigen::Matrix<double, 2, 3>::Zero();
            jacobian_pi(0, 0) = 1. / p_LinC0(2);
            jacobian_pi(0, 2) = -p_LinC0(0) / (p_LinC0(2) * p_LinC0(2));
            jacobian_pi(1, 1) = 1. / p_LinC0(2);
            jacobian_pi(1, 2) = -p_LinC0(1) / (p_LinC0(2) * p_LinC0(2));

            Eigen::Matrix<double, 2, 6> delta;
            const Eigen::Vector3d p_LinC0_optimized =
                T_MtoC0_optimized.transform(p_LinM);

            Eigen::Matrix<double, 3, 6> jacobian_cam_optimized;
            jacobian_cam_optimized.topLeftCorner<3, 3>().setIdentity();
            jacobian_cam_optimized.topRightCorner<3, 3>() =
                - common::skew_x(p_LinC0_optimized);

            delta = jacobian_pi * T_C0toCi[cam_idx].getRotationMatrix() *
                        jacobian_cam_optimized;

            Eigen::Vector3d bearing_homo;
            cameras->getCamera(cam_idx).backProject3(meas, &bearing_homo);
            const double z = bearing_homo(2);
            bearing_homo << bearing_homo(0) / z, bearing_homo(1) / z, 1.0;

            Eigen::Vector2d bearing = bearing_homo.hnormalized();
            Eigen::Vector2d beta = p_LinC0.hnormalized() - bearing;

            H += delta.transpose() * delta;
            g -= delta.transpose() * beta;

            cost_curr += beta.norm();
        }

        if (*cost_init_ptr < 0) {
            *cost_init_ptr = cost_curr;
        }

        Eigen::Matrix<double, 6, 1> epsilon = H.llt().solve(g);
        if (H_ptr != nullptr) {
            *H_ptr = H;
        }
        if (std::isnan(epsilon[0]) || std::isinf(epsilon[0])) {
            break;
        }

        VLOG(4) << "num_iterations " << *num_iterations_ptr <<
                ", curr_cost " << cost_curr <<
                ", epsilon " << epsilon.transpose();

        if (epsilon.norm() < converge_tolerance) {
            is_converged = true;
            VLOG(4) << "Gauss Newton PnP is converged.";
            break;
        }

        // Here we are approximating SE(3) with SO(3) * R(3),
        // assuming left Jacobian of SO(3) is identity for small epsilons.
        T_MtoC0_optimized = common::Exp(epsilon) * T_MtoC0_optimized;
        ++*num_iterations_ptr;

        if (*num_iterations_ptr > max_num_iterations) {
            VLOG(4) << "Reached max iteration numbers.";
            break;
        }
    }

    if (!is_converged && (cost_curr < *cost_init_ptr)) {
        VLOG(4) << "Current cost is better, consider as converged";
        is_converged = true;
    }

    *T_OtoM_final_ptr = T_MtoC0_optimized.inverse() * T_OtoCi[kBaseCamIdx];
    *cost_final_ptr = cost_curr;

    return is_converged;
}

bool RansacP3P(
        const Eigen::Matrix2Xd& keypoints,
        const Eigen::VectorXi& cam_indices,
        const Eigen::Matrix3Xd& p_LinMs,
        const double loop_closure_sigma_pixel,
        const int pnp_num_ransac_iters,
        const aslam::NCamera::ConstPtr& cameras,
        aslam::Transformation* T_OtoM_pnp_ptr,
        std::vector<int>* inliers_ptr,
        std::vector<double>* inlier_distances_to_model_ptr,
        int* num_iters_ptr) {
    CHECK_NOTNULL(T_OtoM_pnp_ptr);
    CHECK_NOTNULL(inliers_ptr);
    CHECK_NOTNULL(inlier_distances_to_model_ptr);
    CHECK_NOTNULL(num_iters_ptr);

    const size_t keypoints_size = static_cast<size_t>(keypoints.cols());
    CHECK_EQ(keypoints_size, static_cast<size_t>(p_LinMs.cols()));
    CHECK_EQ(keypoints_size, cam_indices.rows());

    common::EigenVector3dVec points;
    common::EigenVector3dVec bearings;
    points.resize(keypoints_size);
    bearings.resize(keypoints_size);

    for (size_t i = 0u; i < keypoints_size; ++i) {
        const int camera_index = cam_indices(i);
        cameras->getCamera(camera_index).backProject3(keypoints.col(i), &bearings[i]);
        const double z = bearings[i](2);
        bearings[i] << bearings[i](0) / z, bearings[i](1) / z, 1.0;
        bearings[i].normalize();
        points[i] = p_LinMs.col(i);
    }

    size_t max_num_inliers = 0;
    std::vector<int> vector_inliers;
    std::vector<double> vector_inlier_distances_to_model;
    aslam::Transformation  T_MtoC0_estimated;

    // Add ransac here.
    int current_iterations = 0;
    constexpr int kRansacMinSet = 3;
    int id1, id2, id3;

    const int max_idx = static_cast<int>(keypoints_size) - 1;
    constexpr unsigned seed = 12345u;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> dis(0, max_idx);

    while (current_iterations < pnp_num_ransac_iters) {
        current_iterations++;

        // Get min set of points.
        std::vector<int> index;
        std::set<int> picked_set;
        for(short i = 0; i < kRansacMinSet; ++i) {
            const int idx = dis(mt);
            if (picked_set.count(idx) == 0) {
                index.push_back(idx);
                picked_set.insert(idx);
            }
        }

        id1 = index[0];
        id2 = index[1];
        id3 = index[2];

        common::EigenMatrix34dVec solutions;
        P3PCore(points[id1],
                points[id2],
                points[id3],
                bearings[id1],
                bearings[id2],
                bearings[id3],
                &solutions);

        // Four solutions.
        for (const auto& solution : solutions) {
            Eigen::Quaterniond qgc{solution.block<3, 3> (0, 0)};

            if (std::isnan(qgc.squaredNorm())) {
                continue;
            }

            Eigen::Quaterniond qcg = qgc.conjugate();
            Eigen::Matrix3d rcg = qcg.matrix();
            // The solutions transform point from camera to global here.
            Eigen::Vector3d t = -1.0 * rcg * solution.col(3);
            aslam::Transformation T_MtoC_estimated_temp(qcg, t);
            std::vector<aslam::Transformation> T_MtoC_estimated(1);
            T_MtoC_estimated[kBaseCamIdx] = T_MtoC_estimated_temp;

            std::vector<int> temp_inliers;
            std::vector<double> temp_inlier_distances_to_model;

            CheckInliers(keypoints,
                        cam_indices,
                        p_LinMs,
                        cameras,
                        T_MtoC_estimated,
                        loop_closure_sigma_pixel,
                        &temp_inliers,
                        &temp_inlier_distances_to_model);

            if (temp_inliers.size() > max_num_inliers) {
                max_num_inliers = temp_inliers.size();
                vector_inliers = temp_inliers;
                vector_inlier_distances_to_model =
                    temp_inlier_distances_to_model;
                T_MtoC0_estimated = T_MtoC_estimated[kBaseCamIdx];
            }
        }
    }

    *inliers_ptr = vector_inliers;
    *inlier_distances_to_model_ptr = vector_inlier_distances_to_model;
    *num_iters_ptr = current_iterations;
    const aslam::Transformation& T_OtoC = cameras->get_T_BtoC(kBaseCamIdx);
    *T_OtoM_pnp_ptr = T_MtoC0_estimated.inverse() * T_OtoC;

    return inliers_ptr->size() > kPnPInlierNumThreshold;
}

bool OptimizePnP(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const std::vector<aslam::Transformation>& T_C0toCi,
    const Eigen::Matrix3Xd& p_LinMs,
    const std::vector<int>& inliers_init,
    const aslam::NCamera::ConstPtr& cameras,
    const double converge_tolerance,
    const int max_num_iterations,
    const aslam::Transformation& T_OtoM_init,
    aslam::Transformation* T_OtoM_final_ptr,
    double* cost_init_ptr,
    double* cost_final_ptr,
    int* num_iterations_ptr) {
      CHECK_NOTNULL(T_OtoM_final_ptr);
      CHECK_NOTNULL(cost_init_ptr);
      CHECK_NOTNULL(cost_final_ptr);
      CHECK_NOTNULL(num_iterations_ptr);

      const size_t keypoints_size = keypoints.cols();
      CHECK_EQ(keypoints_size, static_cast<size_t>(p_LinMs.cols()));

      constexpr size_t kMinInliersNumber = 6u;
      if (inliers_init.size() < kMinInliersNumber) {
          return false;
      }

      std::vector<aslam::Transformation> T_OtoCi(T_C0toCi.size());
      for (size_t cam_idx = 0u; cam_idx < T_C0toCi.size(); ++cam_idx) {
          T_OtoCi[cam_idx] = cameras->get_T_BtoC(cam_idx);
      }

      const bool is_converged = OptimizePnPGaussNewton(
          keypoints,
          cam_indices,
          T_C0toCi,
          p_LinMs,
          inliers_init,
          cameras,
          converge_tolerance,
          max_num_iterations,
          T_OtoCi,
          T_OtoM_init,
          T_OtoM_final_ptr,
          nullptr,
          cost_init_ptr,
          cost_final_ptr,
          num_iterations_ptr);

      return is_converged;
}

void CheckInliers(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const Eigen::Matrix3Xd& p_LinMs,
    const aslam::NCamera::ConstPtr& cameras,
    const std::vector<aslam::Transformation>& T_MtoC,
    const double loop_closure_sigma_pixel,
    std::vector<int>* inliers_ptr,
    std::vector<double>* inlier_distances_to_model_ptr) {
    CHECK_GT(loop_closure_sigma_pixel, 0.);

    inliers_ptr->clear();
    inlier_distances_to_model_ptr->clear();

    for (int i = 0; i < keypoints.cols(); ++i) {
        const int camera_index = cam_indices(i);
        Eigen::Vector3d p_LinC =
            T_MtoC[camera_index].transform(p_LinMs.col(i));
        Eigen::Vector2d reprojected_point;

        aslam::ProjectionResult projection_result =
            cameras->getCamera(camera_index).project3(
                p_LinC, &reprojected_point);

        if (projection_result == aslam::ProjectionResult::KEYPOINT_VISIBLE ||
           projection_result ==
                aslam::ProjectionResult::KEYPOINT_OUTSIDE_IMAGE_BOX) {
            const double dist =
                (reprojected_point - keypoints.col(i)).norm() / std::sqrt(2.);

            if (dist < loop_closure_sigma_pixel) {
                inliers_ptr->push_back(i);
                inlier_distances_to_model_ptr->push_back(dist);
            }
        }
    }
}
}
