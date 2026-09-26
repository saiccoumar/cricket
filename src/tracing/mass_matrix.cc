#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/crba.hpp>

namespace cricket
{
    auto trace_mass_matrix(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nv = static_cast<Eigen::Index>(model.nv);

        ADModel ad_model = model.cast<ADCG>();
        ADData ad_data(ad_model);
        ADVectorXs ad_q = ADVectorXs::Zero(static_cast<Eigen::Index>(nq));
        CppAD::Independent(ad_q);

        pinocchio::crba(ad_model, ad_data, ad_q, pinocchio::Convention::WORLD);
        // crba fills the upper triangle; emit the full symmetric matrix, column-major.
        ad_data.M.triangularView<Eigen::StrictlyLower>() =
            ad_data.M.transpose().triangularView<Eigen::StrictlyLower>();
        ADVectorXs ad_m = Eigen::Map<ADVectorXs>(ad_data.M.data(), nv * nv);

        CppAD::ADFun<CGD> func(ad_q, ad_m);
        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(nq);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, static_cast<std::size_t>(nv * nv));
    }
}  // namespace cricket
