#include <vector>
#include <iostream>
#include <iomanip>

#include "CoDiPack/include/codi.hpp"

/// Define a function with its analytical gradient.
class Functional {
protected:
    const double rosenbrock_coeff = 1.22;
public:
    virtual double evaluate_value(std::vector<double> design_variables) const
    {
        double value = 0.0;
        const unsigned int n = design_variables.size();
        // Rosenbrock function
        for (int i = 0; i < n-1; ++i) {
            const double &x0 = design_variables[i];
            const double &x1 = design_variables[i+1];
            value += rosenbrock_coeff*(x1 - x0*x0)*(x1 - x0*x0) + (1.0-x0)*(1.0-x0);
            //value += 2*(x1-x0);
        }

        return value;
    }

    virtual void evaluate_gradient(const std::vector<double> &design_variables, std::vector<double> &gradient) const
    {
        const unsigned int n = design_variables.size();

        for (unsigned int i=1; i<n-1; ++i)
        {
            const double &x = design_variables[i];
            const double &x_m = design_variables[i-1];
            const double &x_p = design_variables[i+1];
            gradient[i] = 2*rosenbrock_coeff*(x-x_m*x_m) - 2*2*rosenbrock_coeff*(x_p - x*x)*x - 2*(1-x);
        }
        auto &x = design_variables;
        gradient[0] = -2*2*rosenbrock_coeff*x[0]*(x[1]-x[0]*x[0]) - 2*(1-x[0]);
        gradient[n-1] = 2*rosenbrock_coeff*(x[n-1]-x[n-2]*x[n-2]);
    }

};

/// Pretend we don't know the analytic formulation, we can always do finite-differences.
class Functional_FiniteDifferences : public Functional
{
public:
    virtual void evaluate_gradient(const std::vector<double> &design_variables, std::vector<double> &gradient) const override
    {
        const double perturbation = 1e-2;
        const unsigned int n = design_variables.size();

        for (unsigned int i=0; i<n; ++i)
        {
            // Positive perturbation
            auto dvar_plus = design_variables;
            dvar_plus[i] += perturbation;
            const double value_plus = evaluate_value(dvar_plus);
            // Negative perturbation
            auto dvar_minus = design_variables;
            dvar_minus[i] -= perturbation;
            const double value_minus = evaluate_value(dvar_minus);

            gradient[i]  = (value_plus - value_minus) / (2.0*perturbation);
        }
    }

};

/// Now we template our function value, this will allow us to automatically differentiate it.
class TemplatedFunctional: public Functional
{

protected:
    template<typename real>
    real value(std::vector<real> design_variables) const
    {
        real value = 0.0;
        const unsigned int n = design_variables.size();
        // Rosenbrock function
        for (int i = 0; i < n-1; ++i) {
            const real &x0 = design_variables[i];
            const real &x1 = design_variables[i+1];
            value += rosenbrock_coeff*(x1 - x0*x0)*(x1 - x0*x0) + (1.0-x0)*(1.0-x0);
            //value += 2*(x1-x0);
        }

        return value;
    }

public:
    virtual double evaluate_value(std::vector<double> design_variables) const override
    {
        return value<double>(design_variables);
    }

    virtual void evaluate_gradient(const std::vector<double> &design_variables, std::vector<double> &gradient) const override
    {
        (void) design_variables;
        (void) gradient;
        std::cout << "Please define evaluate_gradient()" << std::endl;
        std::abort();
    }

};

class TemplatedFunctional_ForwardAD: public TemplatedFunctional
{
public:
    virtual void evaluate_gradient(const std::vector<double> &design_variables, std::vector<double> &gradient) const override
    {
        const unsigned int n = design_variables.size();

        using adtype = codi::RealForward;

        // Copy double values into adtype vector
        std::vector<adtype> design_variables_ad(n);
        for (unsigned int i=0; i<n; ++i) {
            design_variables_ad[i] = design_variables[i];
            design_variables_ad[i].setGradient(0.0);
        }

        // This loop scales with the number of design variables!!!!!!!!!
        for(unsigned int i=0; i<n; ++i) {
            // Set the AD variable to be differentiated against as 1.0
            design_variables_ad[i].setGradient(1.0);

            // Evaluate functional
            adtype functional_value = value<adtype>(design_variables_ad);

            // Extract df/dx_i
            gradient[i] = functional_value.getGradient();
            // Reset AD part to 0.0
            design_variables_ad[i].setGradient(0.0);
        }
    }

};

class TemplatedFunctional_ReverseAD: public TemplatedFunctional
{
public:
    virtual void evaluate_gradient(const std::vector<double> &design_variables, std::vector<double> &gradient) const override
    {
        const unsigned int n = design_variables.size();

        using adtype = codi::RealReverse;

        // Activate the tape
        codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
        tape.setActive();

        // Copy double values into adtype vector
        std::vector<adtype> design_variables_ad(n);
        for (unsigned int i=0; i<n; ++i) {
            design_variables_ad[i] = design_variables[i];
            // Mark inputs
            tape.registerInput(design_variables_ad[i]);
        }

        // Evaluate functional
        adtype functional_value = value<adtype>(design_variables_ad);

        // Mark output
        tape.registerOutput(functional_value);

        // Turn off taping
        tape.setPassive();

        // Set the left value to be 1.0
        functional_value.setGradient(1.0);

        // Evaluate the tape
        tape.evaluate();

        // Extract df/dx
        for (unsigned int i=0; i<n; ++i) {
            gradient[i] = design_variables_ad[i].getGradient();
        }
    }

};

double norm(std::vector<double> vec) {
    double norm = 0;
    for (auto &v : vec) {
        norm += v*v;
    }
    return sqrt(norm);
}

class GradientDescent
{
public:
    static void optimize(const Functional &f, const std::vector<double> &initial_design, std::vector<double> &final_design,
                         const double tolerance = 1e-5, const unsigned int max_iterations = 200)
    {
        std::cout << "Initial design: ";
        for (const auto &v : initial_design) {
            std::cout << v << "\t";
        }
        const unsigned int n = initial_design.size();

        // Initialize current iteration state
        std::vector<double> current_design = initial_design;
        std::vector<double> old_design(n), descent_direction(n), gradient(n);

        // Evaluate value and gradient
        f.evaluate_gradient(current_design, gradient);
        double value         = f.evaluate_value(current_design);
        double gradient_norm = norm(gradient);

        std::cout << "\t" << " Iteration "     << 0 << "\t" << " Value     "     << value << "\t" << " Gradient norm " << gradient_norm << std::endl;

        for (unsigned int iteration = 0;
                gradient_norm > tolerance && iteration < max_iterations;
                ++iteration)
        {
            // Compute descent direction
            for (int i=0; i<n; ++i) {
                descent_direction[i] = -1.0 * gradient[i];
            }

            // Line search and function evaluation
            double step_length = 1.0;
            double old_value = value;
            old_design = current_design;
            value = 1e300;
            while (value > old_value) {
                for (int i=0; i<n; ++i) {
                    current_design[i] = old_design[i] + step_length*descent_direction[i];
                }
                value    = f.evaluate_value(current_design);
                step_length *= 0.5;
            }

            // Evaluate gradient
            f.evaluate_gradient(current_design, gradient);
            gradient_norm = norm(gradient);

            if ((iteration + 1) % 10 == 0) std::cout << "\t" << " Iteration "     << iteration + 1 << "\t" << " Value     "     << value << "\t" << " Gradient norm " << gradient_norm << std::endl;
        }

        final_design = current_design;
        std::cout << "Final design: ";
        for (const auto &v : final_design) {
            std::cout << v << "\t";
        }
        std::cout << std::endl;
        
    }
};
int main () {
    std::vector<double> initial_design(3, 0.5);
    std::vector<double> final_design;

    Functional rosenbrock;
    Functional_FiniteDifferences rosenbrock_fd;
    TemplatedFunctional_ReverseAD rosenbrock_rad;
    TemplatedFunctional_ForwardAD rosenbrock_fad;

    std::vector<double> gradient = initial_design;

    std::cout << std::scientific << std::setprecision(13);
    rosenbrock.evaluate_gradient(initial_design, gradient);
    std::cout << " Analytical gradients..." << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
    rosenbrock_fd.evaluate_gradient(initial_design, gradient);
    std::cout << " Finite-dif gradients..." << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
    rosenbrock_rad.evaluate_gradient(initial_design, gradient);
    std::cout << " Forward-AD gradients..." << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
    rosenbrock_fad.evaluate_gradient(initial_design, gradient);
    std::cout << " Reverse-AD gradients..." << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;

    std::cout << std::scientific << std::setprecision(3);
    std::cout << std::endl << " Optimize using analytical gradients..." << std::endl;
    GradientDescent::optimize(rosenbrock, initial_design, final_design);

    std::cout << std::endl << " Optimize using finite differences..." << std::endl;
    GradientDescent::optimize(rosenbrock_fd, initial_design, final_design);

    std::cout << std::endl << " Optimize using reverse automatic differentiation..." << std::endl;
    GradientDescent::optimize(rosenbrock_rad, initial_design, final_design);

    std::cout << std::endl << " Optimize using forward automatic differentiation..." << std::endl;
    GradientDescent::optimize(rosenbrock_fad, initial_design, final_design);
}
