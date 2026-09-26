#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace cricket
{
    auto trace_frames(const RobotInfo &info, const std::vector<std::string> &frames, const std::string &language)
        -> Traced
    {
        std::vector<pinocchio::FrameIndex> ids;
        ids.reserve(frames.size());
        for (const auto &name : frames)
        {
            // URDF joints and links may share a name; the joint frame wins.
            const auto joint_types = static_cast<pinocchio::FrameType>(pinocchio::JOINT | pinocchio::FIXED_JOINT);
            if (info.model.existFrame(name, joint_types))
            {
                ids.push_back(info.model.getFrameId(name, joint_types));
            }
            else if (info.model.existFrame(name))
            {
                ids.push_back(info.model.getFrameId(name));
            }
            else
            {
                throw std::runtime_error(fmt::format("cricket::trace_frames: unknown frame '{}'", name));
            }
        }

        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto n_out = 12 * ids.size();

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        ADVectorXs ad_q = ADVectorXs::Zero(info.model.nq);
        CppAD::Independent(ad_q);

        pinocchio::forwardKinematics(ad_model, ad_data, ad_q);
        pinocchio::updateFramePlacements(ad_model, ad_data);

        ADVectorXs data(n_out);
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            const auto &oMf = ad_data.oMf[ids[i]];
            const auto &R = oMf.rotation();
            const auto k = 12 * i;
            data[k + 0] = oMf.translation()[0];
            data[k + 1] = oMf.translation()[1];
            data[k + 2] = oMf.translation()[2];
            // Eigen stores as column major
            for (auto c = 0; c < 3; ++c)
            {
                for (auto r = 0; r < 3; ++r)
                {
                    data[k + 3 + 3 * c + r] = R(r, c);
                }
            }
        }

        CppAD::ADFun<CGD> frames_func(ad_q, data);

        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(nq);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = frames_func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, n_out);
    }
}  // namespace cricket
