#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "cppad/cg/lang/c/language_c.hpp"

namespace CppAD
{
    namespace cg
    {
        // Single-precision C for CUDA device code: literals get an `f` suffix and math calls use the
        // float overloads, so nothing silently promotes to double (1/32 FP32 throughput on consumer GPUs).
        template <class Base>
        class LanguageCUDA : public LanguageC<Base>
        {
        public:
            explicit LanguageCUDA(size_t spaces = 3) : LanguageC<Base>("float", spaces)
            {
            }

            virtual void printParameter(const Base &value)
            {
                writeParameter(value, LanguageC<Base>::_code);
            }

            virtual void pushParameter(const Base &value)
            {
                writeParameter(value, LanguageC<Base>::_streamStack);
            }

            template <class Output>
            void writeParameter(const Base &value, Output &output)
            {
                std::ostringstream os;
                os << std::setprecision(9) << value;  // round-trips float32

                std::string number = os.str();
                output << number;

                if (number.find('.') == std::string::npos and number.find('e') == std::string::npos)
                {
                    output << '.';
                }
                output << 'f';
            }

#define CRICKET_CUDA_FUNCNAME(fn, name)                                                                      \
    const std::string &fn##FuncName() override                                                               \
    {                                                                                                        \
        static const std::string s(name);                                                                    \
        return s;                                                                                            \
    }

            CRICKET_CUDA_FUNCNAME(abs, "fabsf")
            CRICKET_CUDA_FUNCNAME(acos, "acosf")
            CRICKET_CUDA_FUNCNAME(asin, "asinf")
            CRICKET_CUDA_FUNCNAME(atan, "atanf")
            CRICKET_CUDA_FUNCNAME(cos, "cosf")
            CRICKET_CUDA_FUNCNAME(cosh, "coshf")
            CRICKET_CUDA_FUNCNAME(exp, "expf")
            CRICKET_CUDA_FUNCNAME(log, "logf")
            CRICKET_CUDA_FUNCNAME(pow, "powf")
            CRICKET_CUDA_FUNCNAME(sin, "sinf")
            CRICKET_CUDA_FUNCNAME(sinh, "sinhf")
            CRICKET_CUDA_FUNCNAME(sqrt, "sqrtf")
            CRICKET_CUDA_FUNCNAME(tan, "tanf")
            CRICKET_CUDA_FUNCNAME(tanh, "tanhf")

#undef CRICKET_CUDA_FUNCNAME
        };
    }  // namespace cg
}  // namespace CppAD
