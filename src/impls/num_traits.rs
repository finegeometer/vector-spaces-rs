use crate::{Space, Linear};
use nalgebra::{Scalar, DefaultAllocator, allocator::Allocator};
use alga::general::{ClosedAdd, ClosedMul, ComplexField};
use num_traits::*;

impl<N: Scalar + Zero + ClosedAdd, A: Space, B: Space> Zero for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>
{
	fn zero() -> Self {
		Self(Zero::zero())
	}
	fn is_zero(&self) -> bool {
		self.0.is_zero()
	}
}

impl<N: Scalar + Zero + One + ClosedAdd + ClosedMul, A: Space> One for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>
{
	fn one() -> Self {
		Self(One::one())
	}
	fn is_one(&self) -> bool {
		self.0.is_one()
	}
}

impl<N: Scalar + ComplexField, A: Space, B: Space<Dim = A::Dim>> Inv for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
	DefaultAllocator: Allocator<N, A::Dim, B::Dim>
{
	type Output = Linear<N, B, A>;
	fn inv(self) -> Linear<N, B, A> {
		Self(self.0.try_inverse().unwrap_or_else(zero))
	}
}