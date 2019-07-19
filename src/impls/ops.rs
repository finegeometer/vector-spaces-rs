use crate::{Space, Linear};
use core::ops::*;
use nalgebra::{Scalar, DefaultAllocator, allocator::Allocator};
use num_traits::{Zero, One};
use alga::general::{ClosedAdd, ClosedSub, ClosedNeg, ClosedMul};

impl<N: Scalar + ClosedAdd, A: Space, B: Space> Add for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Output = Self;
	fn add(self, other: Self) -> Self {
		Self(self.0 + other.0)
	}
}

impl<N: Scalar + ClosedAdd, A: Space, B: Space> AddAssign for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn add_assign(&mut self, other: Self) {
		self.0 += other.0
	}
}

impl<N: Scalar + ClosedNeg, A: Space, B: Space> Neg for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Output = Self;
	fn neg(self) -> Self {
		Self(-self.0)
	}
}

impl<N: Scalar + ClosedSub, A: Space, B: Space> Sub for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Output = Self;
	fn sub(self, other: Self) -> Self {
		Self(self.0 - other.0)
	}
}

impl<N: Scalar + ClosedSub, A: Space, B: Space> SubAssign for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn sub_assign(&mut self, other: Self) {
		self.0 -= other.0
	}
}

impl<N: Scalar + ClosedAdd + ClosedMul, A: Space, B: Space> Mul<N> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Output = Self;
	fn mul(self, a: N) -> Self {
		Self(self.0 * a)
	}
}

impl<N: Scalar + ClosedAdd + ClosedMul, A: Space, B: Space> MulAssign<N> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn mul_assign(&mut self, a: N) {
		self.0 *= a
	}
}

impl<N: Scalar + Zero + One + ClosedAdd + ClosedMul, A: Space, B: Space, C: Space> Mul<Linear<N, A, B>> for Linear<N, B, C> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
	DefaultAllocator: Allocator<N, C::Dim, A::Dim>,
	DefaultAllocator: Allocator<N, C::Dim, B::Dim>,
{
	type Output = Linear<N, A, C>;
	fn mul(self, other: Linear<N, A, B>) -> Linear<N, A, C> {
		Self(self.0 * other.0)
	}
}

impl<N: Scalar + Zero + One + ClosedAdd + ClosedMul, A: Space, B: Space> MulAssign<Linear<N, A, A>> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{
	fn mul_assign(&mut self, other: Linear<N, A, A>) {
		self.0 *= other.0
	}
}


