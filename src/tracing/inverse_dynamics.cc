#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/rnea.hpp>

namespace cricket
{
    auto trace_inverse_dynamics(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nv = static_cast<std::size_t>(model.nv);
        const auto n_input = nq + 2 * nv;

        ADModel ad_model = model.cast<ADCG>();
        ADData ad_data(ad_model);
        ADVectorXs ad_input = ADVectorXs::Zero(static_cast<Eigen::Index>(n_input));
        CppAD::Independent(ad_input);

        ADVectorXs ad_tau = pinocchio::rnea(
            ad_model, ad_data, ad_input.head(nq), ad_input.segment(nq, nv), ad_input.tail(nv));

        CppAD::ADFun<CGD> func(ad_input, ad_tau);
        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(n_input);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, nv);
    }
}  // namespace cricket
