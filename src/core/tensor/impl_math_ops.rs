use std::{
    fmt::Display,
    ops::{self, Neg},
};

use num::Complex;
use num_traits::{float::FloatConst, Float, Inv, NumCast, Pow, Signed};

use crate::utils::{
    errors::Errors,
    math::{
        erf::{erf, erfc},
        expit::expit,
        log_gamma::{log_gamma, log_gamma_float},
        sinc::sinc,
        xlogy::xlogy, gamma::{gamma_float, gamma}, logit::logit,
    },
};

use super::tensor::Tensor;

impl<T: Copy> Tensor<T> {
    /// Apply any function to every element of self and create a new tensor with the results
    /// This is similar to rust's .map() in iterators
    ///
    /// # Arguments
    /// * f - Function that takes type T and returns a new value R
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor
    /// let some_complicated_function_values_as_tensor = t.map(|x| some_complicated_function(x));
    /// ```
    pub fn map<R: Copy, F: Fn(T) -> R>(&self, f: F) -> Tensor<R> {
        let res: Vec<_> = self.into_iter().map(|x| f(x)).collect();
        Tensor::from_slice_and_dims(&res, &self.dims).unwrap()
    }

    /// elements) and create a new tensor with the results
    /// This is similar to rust's .zip().map() in iterators
    ///
    /// # Arguments
    /// * other - The other tensor
    /// * f - Function that takes type T and returns a new value R
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor
    /// let dist_from_origin_tensor = x_coords.map_with(&y_coords, |(x, y)| (x * x + y * y).sqrt());
    /// ```
    pub fn map_with<O: Copy, R: Copy, F: Fn(T, O) -> R>(
        &self,
        other: &Tensor<O>,
        f: F,
    ) -> Tensor<R> {
        let res: Vec<_> = self
            .into_iter()
            .zip(other.into_iter())
            .map(|(x, y)| f(x, y))
            .collect();
        Tensor::from_slice_and_dims(&res, &self.dims).unwrap()
    }
}

impl<T: Copy + Signed> Tensor<T> {
    /// Calculate the absolute function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let abs_t = t.abs();
    /// ```
    pub fn abs(&self) -> Tensor<T> {
        self.map(|x| x.abs())
    }

    /// Calculate the absolute function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let absolute_t = t.absolute();
    /// ```
    pub fn absolute(&self) -> Tensor<T> {
        self.abs()
    }
}

impl<T: Copy + Float> Tensor<T> {
    /// Calculate the inverse sine function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let asin_t = t.asin();
    /// ```
    pub fn asin(&self) -> Tensor<T> {
        self.map(|x| x.asin())
    }

    /// Calculate the inverse cosine function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let acos_t = t.acos();
    /// ```
    pub fn acos(&self) -> Tensor<T> {
        self.map(|x| x.acos())
    }

    /// Calculate the inverse tangent function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Examples
    /// ```rust
    /// let atan_t = t.atan();
    /// ```
    pub fn atan(&self) -> Tensor<T> {
        self.map(|x| x.atan())
    }

    /// Calculate the inverse tangent 2 function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Arguments
    /// * other - The other tensor with which the atan2 function is to be calculated
    ///
    /// # Examples
    /// ```rust
    /// let atan2_t = rise_t.atan2(run_t);
    /// ```
    pub fn atan2(&self, other: &Tensor<T>) -> Tensor<T> {
        self.map_with(other, |x, y| x.atan2(y))
    }

    /// Calculate the inverse hyperbolic sine function for every element of self and returns a new
    /// tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let asinh_t = t.asinh();
    /// ```
    pub fn asinh(&self) -> Tensor<T> {
        self.map(|x| x.asinh())
    }

    /// Calculate the inverse hyperbolic cosine function for every element of self and returns a
    /// new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let acosh_t = t.acosh();
    /// ```
    pub fn acosh(&self) -> Tensor<T> {
        self.map(|x| x.acosh())
    }

    /// Calculate the inverse hyperbolic tangent function for every element of self and returns a
    /// new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let atanh_t = t.atanh();
    /// ```
    pub fn atanh(&self) -> Tensor<T> {
        self.map(|x| x.atanh())
    }

    /// Calculate the inverse sine function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let arcsin_t = t.arcsin();
    /// ```
    pub fn arcsin(&self) -> Tensor<T> {
        self.asin()
    }

    /// Calculate the inverse cosine function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let arccos_t = t.arccos();
    /// ```
    pub fn arccos(&self) -> Tensor<T> {
        self.acos()
    }

    /// Calculate the inverse tangent function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Examples
    /// ```rust
    /// let arctan_t = t.arctan();
    /// ```
    pub fn arctan(&self) -> Tensor<T> {
        self.atan()
    }

    /// Calculate the inverse tangent 2 function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Arguments
    /// * other - The other tensor with which the atan2 function is to be calculated
    ///
    /// # Examples
    /// ```rust
    /// let arctan2_t = t.arctan2();
    /// ```
    pub fn arctan2(&self, other: &Tensor<T>) -> Tensor<T> {
        self.atan2(other)
    }

    /// Calculate the inverse hyperbolic sine function for every element of self and returns a new
    /// tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let arcsinh_t = t.arcsinh();
    /// ```
    pub fn arcsinh(&self) -> Tensor<T> {
        self.asinh()
    }

    /// Calculate the inverse hyperbolic cosine function for every element of self and returns a
    /// new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let arccosh_t = t.arccosh();
    /// ```
    pub fn arccosh(&self) -> Tensor<T> {
        self.acosh()
    }

    /// Calculate the inverse hyperbolic tangent function for every element of self and returns a
    /// new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let arctanh_t = t.arctanh();
    /// ```
    pub fn arctanh(&self) -> Tensor<T> {
        self.atanh()
    }

    /// Calculate the ceiling function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let ceil_t = t.ceil();
    /// ```
    pub fn ceil(&self) -> Tensor<T> {
        self.map(|x| x.ceil())
    }

    /// Calculate the cosine function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let cos_t = t.cos();
    /// ```
    pub fn cos(&self) -> Tensor<T> {
        self.map(|x| x.cos())
    }

    /// Calculate the hyperbolic cosine function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Examples
    /// ```rust
    /// let cosh_t = t.cosh();
    /// ```
    pub fn cosh(&self) -> Tensor<T> {
        self.map(|x| x.cosh())
    }

    /// Convert Degrees to Radians for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let degrees_to_radians_t = t.degrees_to_radians();
    /// ```
    pub fn degrees_to_radians(&self) -> Tensor<T> {
        self.map(|x| x.to_radians())
    }

    /// Convert radians to degrees for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let radians_to_degrees_t = t.radians_to_degrees();
    /// ```
    pub fn radians_to_degrees(&self) -> Tensor<T> {
        self.map(|x| x.to_degrees())
    }

    /// Calculate the exponential function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let exp_t = t.exp();
    /// ```
    pub fn exp(&self) -> Tensor<T> {
        self.map(|x| x.exp())
    }

    /// Calculate the exponential function with base 2 for every element of self and returns a new
    /// tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let exp_2_t = t.exp_2();
    /// ```
    pub fn exp_2(&self) -> Tensor<T> {
        self.map(|x| x.exp2())
    }

    /// Truncate every element of self and returns a new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let trunc_t = t.trunc();
    /// ```
    pub fn trunc(&self) -> Tensor<T> {
        self.map(|x| x.trunc())
    }

    /// Calculate the natura logarithm function for every element of self and returns a new tensor
    /// of the results
    ///
    /// # Examples
    /// ```rust
    /// let log_t = t.log();
    /// ```
    pub fn log(&self) -> Tensor<T> {
        self.map(|x| x.ln())
    }

    /// Calculate the logarithm function with base 10 for every element of self and returns a new
    /// tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let log_10_t = t.log_10();
    /// ```
    pub fn log_10(&self) -> Tensor<T> {
        self.map(|x| x.log10())
    }

    /// Calculate the logarithm function with base 2 for every element of self and returns a new
    /// tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let log_2_t = t.log_2();
    /// ```
    pub fn log_2(&self) -> Tensor<T> {
        self.map(|x| x.log2())
    }

    /// Round every element of self and returns a new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let round_t = t.round();
    /// ```
    pub fn round(&self) -> Tensor<T> {
        self.map(|x| x.round())
    }

    /// Calculate the floor function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let floor_t = t.floor();
    /// ```
    pub fn floor(&self) -> Tensor<T> {
        self.map(|x| x.floor())
    }

    /// Get fractional complement for every element of self and returns a new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let frac_t = t.frac();
    /// ```
    pub fn frac(&self) -> Tensor<T> {
        self.map(|x| x.fract())
    }

    /// Calculate the square root function for every element of self and returns a new tensor of
    /// the results
    ///
    /// # Examples
    /// ```rust
    /// let sqrt_t = t.sqrt();
    /// ```
    pub fn sqrt(&self) -> Tensor<T> {
        self.map(|x| x.sqrt())
    }

    /// Finds length of hypotenuse of right triangle with sides lengths as elements in self and
    /// returns a new tensor of the results
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let hypot_t = t.hypot();
    /// ```
    pub fn hypot(&self, other: &Tensor<T>) -> Tensor<T> {
        self.map_with(other, |x, y| x.hypot(y))
    }

    /// Calculate the reciprocal of the square root function for every element of self and returns
    /// a new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let rsqrt_t = t.rsqrt();
    /// ```
    pub fn rsqrt(&self) -> Tensor<T> {
        // TODO: Use Fast Inverse sqrt algorithm
        self.map(|x| x.sqrt().recip())
    }

    /// Here we calculate expit(x) for every element in `self`
    ///
    /// The Expit Function, also known as the logistic function, or the sigmoid function is a S shaped
    /// curve that appears frequently in many branches of science.
    ///
    /// Formally it is defined as:
    /// sigmoid(x) = 1 / (1 + e^(-x))
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let expit_1 = t.expit();
    /// ```
    pub fn expit(&self) -> Tensor<T> {
        self.map(|x| expit(x))
    }

    /// Here we calculate logit(x) for every element in `self`
    ///
    /// The Logit function is the quantile function associated with the standard logistic distribution.
    ///
    /// Formally it is defined as:
    /// logit(p) = ln(p / (1 - p))
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let logit_1 = t.logit();
    /// ```
    pub fn logit(&self) -> Tensor<T> {
        self.map(|x| logit(x))
    }

    /// Here we calculate xlogy(x) for every element in `self`
    ///
    /// The xlogy function is formally defined as:
    /// xlog(y) = x * log(y)
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let xlogy_1 = t.xlogy();
    /// ```
    pub fn xlogy(&self, other: &Tensor<T>) -> Tensor<T> {
        self.map_with(other, |x, y| xlogy(x, y))
    }
}

impl<T: Copy + Float + NumCast> Tensor<T> {
    /// Here we calculate erf(x) for every element in `self`
    ///
    /// The error function, denoted by erf, is encountered when integrating the normal distribution.
    ///
    /// It is formally defined as a complex valued function:
    /// erf(z) = 2 / sqrt(pi)  * integrate(e^-(t^2) dt) from 0 to z.
    ///
    /// Here we approximate erf(z) for every element in `self` only for real z with maximum error
    /// upto 1.5 * 10^(-7).
    ///
    /// The approximation method can be found on the wikipedia article of the erf function.
    /// This approximation method is credited to Abramowitz and Stegun.
    ///
    /// See the [wikipedia article](https://en.wikipedia.org/wiki/Error_function) for more information
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_erf = t.erf();
    /// ```
    pub fn erf(&self) -> Tensor<T> {
        self.map(|x| erf(x))
    }

    /// Here we calculate erfc(x) for every element in `self`
    ///
    /// The error function complement, denoted by erfc, is encountered when integrating the normal
    /// distribution.
    ///
    /// It is formally defined as a complex valued function:
    /// erfc(z) = 1 - erf(z) = 1 - 2 / sqrt(pi)  * integrate(e^-(t^2) dt) from 0 to z.
    ///
    /// Here we approximate erfc(z) for every element in `self` only for real z with maximum error
    /// upto 1.5 * 10^(-7).
    ///
    /// The approximation method can be found on the wikipedia article of the erfc function.
    /// This approximation method is credited to Abramowitz and Stegun.
    ///
    /// See the [wikipedia article](https://en.wikipedia.org/wiki/Error_function) for more information
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_erfc = t.erfc();
    /// ```
    pub fn erfc(&self) -> Tensor<T> {
        self.map(|x| erfc(x))
    }
}

impl<T: Copy + Float + FloatConst> Tensor<T> {
    /// Here we calculate the normalized sinc function for every element in `self`
    ///
    /// The normalized sinc function is defined as the Fourier transform of the rectangular
    /// function with no scaling.
    ///
    /// It is formally defined as:
    /// sinc(x) = sin(pi * x) / (pi * x)
    /// where x != 0, and sinc(0) = 1
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_sinc = t.sinc();
    /// ```
    pub fn sinc(&self) -> Tensor<T> {
        self.map(|x| sinc(x))
    }

    /// Here we calculate log_gamma(x) for every element in `self` for real x
    ///
    /// The log gamma function is defined as the natural log of the gamma function, which is the
    /// most used extension to the factorial function into the complex plane.
    ///
    /// It is formally defined as a complex valued function:
    /// log_gamma(x) = ln(gamma(x)) = ln( integrate(t^(z - 1) * e^(-t) * dt) from 0 to +inf )
    ///
    /// Here we approximate log_gamma(x) with accuracy > 10 d.p
    ///
    /// The approximation is based on the implementation of SciPy.
    /// For small values (< 0.2), this returns the Taylor series expansion of log gamma around 0
    /// with 23 terms.
    /// For slightly larger values, this returns the result using a recursive formula and
    /// the Taylor series expansion like before.
    /// For big values (> 7), this returns the famous stirling approximation for log gamma
    /// For values in between, this returns the result by repeatedly applying another recursive.
    /// formula, and finally using the stirling approximation.
    /// For negative values, this returns the result using a reflection formula.
    ///
    /// See
    /// [scipy's implementation](https://github.com/scipy/scipy/blob/main/scipy/special/_loggamma.pxd)
    /// and Hare "Computing the Principal Branch of log-Gamma" on Journal of Algorithms 1997 for
    /// more details
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_gamma = t.log_gammaf();
    /// ```
    pub fn log_gammaf(&self) -> Tensor<T> {
        self.map(|x| log_gamma_float(x))
    }

    /// Calculate Gamma Function (for reals)
    ///
    /// The gamma function is the most used extension to the factorial function into the complex
    /// plane.
    ///
    /// It is formally defined as a complex valued function:
    /// gamma(x) = integrate(t^(z - 1) * e^(-t) * dt) from 0 to +inf
    ///
    /// Here we approximate gamma(x) with accuracy > 10 d.p
    ///
    /// Under the hood, this exponentiates log_gamma(x)
    /// See [Tensor::log_gamma] for details of the approximation.
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_gamma = t.gammaf();
    /// ```
    pub fn gammaf(&self) -> Tensor<T> {
        self.map(|x| gamma_float(x))
    }
}

impl<T: Copy + Float + FloatConst> Tensor<Complex<T>> {
    /// Here we calculate log_gamma(x) for every element in `self` for complex x
    ///
    /// The log gamma function is defined as the natural log of the gamma function, which is the
    /// most used extension to the factorial function into the complex plane.
    ///
    /// It is formally defined as a complex valued function:
    /// log_gamma(x) = ln(gamma(x)) = ln( integrate(t^(z - 1) * e^(-t) * dt) from 0 to +inf )
    ///
    /// Here we approximate log_gamma(x) with accuracy > 10 d.p
    ///
    /// The approximation is based on the implementation of SciPy.
    /// For small values (< 0.2), this returns the Taylor series expansion of log gamma around 0
    /// with 23 terms.
    /// For slightly larger values, this returns the result using a recursive formula and
    /// the Taylor series expansion like before.
    /// For big values (> 7), this returns the famous stirling approximation for log gamma
    /// For values in between, this returns the result by repeatedly applying another recursive.
    /// formula, and finally using the stirling approximation.
    /// For negative values, this returns the result using a reflection formula.
    ///
    /// See
    /// [scipy's implementation](https://github.com/scipy/scipy/blob/main/scipy/special/_loggamma.pxd)
    /// and Hare "Computing the Principal Branch of log-Gamma" on Journal of Algorithms 1997 for
    /// more details
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_gamma = t.log_gamma();
    /// ```
    pub fn log_gamma(&self) -> Tensor<Complex<T>> {
        self.map(|x| log_gamma(x))
    }

    /// Calculate Gamma Function (for complex)
    /// The gamma function is the most used extension to the factorial function into the complex
    /// plane.
    /// It is formally defined as a complex valued function:
    /// gamma(x) = integrate(t^(z - 1) * e^(-t) * dt) from 0 to +inf
    ///
    /// Here we approximate gamma(x) with accuracy > 10 d.p
    ///
    /// Under the hood, this exponentiates log_gamma(x)
    /// See [`log_gamma`] for details of the approximation.
    ///
    /// # Examples
    /// ```rust
    /// // t is some tensor of floats
    /// let t_gamma = t.gammaf();
    /// ```
    pub fn gamma(&self) -> Tensor<Complex<T>> {
        self.map(|x| gamma(x))
    }
}

impl<R: Copy, T: Copy + ops::Mul<Output = R>> Tensor<T> {
    /// Calculate the square function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let square_t = t.square();
    /// ```
    pub fn square(&self) -> Tensor<R> {
        self.map(|x| x * x)
    }
}

impl<T: Copy> Tensor<T> {
    /// Pairwise add elements from self and other and return tensor with results
    ///
    /// # Examples
    /// ```rust
    /// let a_plus_b = a.element_add(b);
    /// ```
    pub fn element_add<O: Copy, R: Copy>(&self, other: &Tensor<O>) -> Tensor<R>
    where
        T: ops::Add<O, Output = R>,
    {
        self.map_with(other, |x, y| x + y)
    }

    /// Pairwise subtract elements from self and other and return tensor with results
    ///
    /// # Examples
    /// ```rust
    /// let a_plus_b = a.element_add(b);
    /// ```
    pub fn element_sub<O: Copy, R: Copy>(&self, other: &Tensor<O>) -> Tensor<R>
    where
        T: ops::Sub<O, Output = R>,
    {
        self.map_with(other, |x, y| x - y)
    }

    /// Pairwise multiply elements from self and other and return tensor with results
    ///
    /// # Examples
    /// ```rust
    /// let a_plus_b = a.element_add(b);
    /// ```
    pub fn element_mul<O: Copy, R: Copy>(&self, other: &Tensor<O>) -> Tensor<R>
    where
        T: ops::Mul<O, Output = R>,
    {
        self.map_with(other, |x, y| x * y)
    }

    /// Pairwise divide elements from self and other and return tensor with results
    ///
    /// # Examples
    /// ```rust
    /// let a_plus_b = a.element_add(b);
    /// ```
    pub fn element_div<O: Copy, R: Copy>(&self, other: &Tensor<O>) -> Tensor<R>
    where
        T: ops::Div<O, Output = R>,
    {
        self.map_with(other, |x, y| x / y)
    }
}

impl<T: Copy + PartialOrd + Display> Tensor<T> {
    /// Clamp every element in self between mn, and mx and returns tensor of the results
    ///
    /// clamp(x) = max(mn, min(mx, x))
    ///
    /// # Arguments
    /// * mn - Minimum for clamp
    /// * mx - Maximum for clamp
    ///
    /// # Examples
    /// ```rust
    /// let clamp_t = t.clamp();
    /// ```
    pub fn clamp(&self, mn: T, mx: T) -> Result<Tensor<T>, Errors> {
        // TODO: TESTS!
        if mn > mx {
            return Err(Errors::InputError(format!(
                "Clamp Error, expected mn <= mx, found {mn} > {mx}"
            )));
        }
        Ok(self.map(|x| num_traits::clamp(x, mn, mx)))
    }

    /// Alias to [Tensor::clamp]
    pub fn clip(&self, mn: T, mx: T) -> Result<Tensor<T>, Errors> {
        self.clamp(mn, mx)
    }
}

impl<T: Copy + Pow<T, Output = T>> Tensor<T> {
    /// Calculate the exponential function for every element of self with exponent as the
    /// corresponding element in other and returns a new tensor of the results
    ///
    /// # Examples
    /// ```rust
    /// let pow_t = t.pow();
    /// ```
    pub fn pow(&self, other: &Tensor<T>) -> Tensor<T> {
        self.map_with(other, |x, y| x.pow(y))
    }
}

impl<T: Copy + Neg<Output = T>> Tensor<T> {
    /// Calculate the Negative function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let neg_t = t.neg();
    /// ```
    pub fn neg(&self) -> Tensor<T> {
        self.map(|x| x.neg())
    }

    /// Calculate the Negative function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let negative_t = t.negative();
    /// ```
    pub fn negative(&self) -> Tensor<T> {
        self.neg()
    }
}

impl<T: Copy + Inv<Output = T>> Tensor<T> {
    /// Calculate the reciprocal function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let inv_t = t.inv();
    /// ```
    pub fn inv(&self) -> Tensor<T> {
        self.map(|x| x.inv())
    }

    /// Calculate the reciprocal function for every element of self and returns a new tensor of the
    /// results
    ///
    /// # Examples
    /// ```rust
    /// let reciprocal_t = t.reciprocal();
    /// ```
    pub fn reciprocal(&self) -> Tensor<T> {
        self.inv()
    }
}
