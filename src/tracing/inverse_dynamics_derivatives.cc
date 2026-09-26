#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace cricket
{
    auto trace_inverse_dynamics_derivatives(const pinocchio::Model &model, const std::string &language)
        -> Traced
    {
        // Pinocchio only asserts this (compiled out in release builds), and silently gives wrong
        // derivatives for mimic joints; refuse instead.
        if (not model.check(pinocchio::MimicChecker()))
        {
            throw std::runtime_error(
                "cricket::trace_inverse_dynamics_derivatives: models with mimic joints are not supported");
        }
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nv = static_cast<std::size_t>(model.nv);
        const auto n_input = nq + 2 * nv;
        const auto n = static_cast<Eigen::Index>(nv);

        ADModel ad_model = model.cast<ADCG>();
        ADData ad_data(ad_model);
        ADVectorXs ad_input = ADVectorXs::Zero(static_cast<Eigen::Index>(n_input));
        CppAD::Independent(ad_input);

        // Pinocchio's analytical derivatives (not AD of RNEA): far smaller traces.
        using ADMatrix = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;
        ADMatrix dtau_dq = ADMatrix::Zero(n, n), dtau_dv = ADMatrix::Zero(n, n), dtau_da = ADMatrix::Zero(n, n);
        pinocchio::computeRNEADerivatives(
            ad_model, ad_data, ad_input.head(nq), ad_input.segment(nq, nv), ad_input.tail(nv), dtau_dq, dtau_dv,
            dtau_da);

        // [dtau/dq | dtau/dv], each n x n column-major.
        ADVectorXs ad_out(2 * n * n);
        ad_out << Eigen::Map<ADVectorXs>(dtau_dq.data(), n * n), Eigen::Map<ADVectorXs>(dtau_dv.data(), n * n);

        CppAD::ADFun<CGD> func(ad_input, ad_out);
        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(n_input);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, 2 * nv * nv);
    }
}  // namespace cricket
