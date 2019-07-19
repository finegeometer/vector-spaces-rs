use crate::{Space, Linear};
use nalgebra::{Scalar, DefaultAllocator, allocator::Allocator};
use alga::{general::*, linear::VectorSpace};
use num_traits::{Zero, One, Inv};


impl<N: Scalar + Zero, A: Space, B: Space> Identity<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn identity() -> Self {
		Self(Identity::<Additive>::identity())
	}
}

impl<N: Scalar + ClosedAdd, A: Space, B: Space> AbstractMagma<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn operate(&self, other: &Self) -> Self {
		Self(self.0.op(Additive, &other.0))
	}
}


impl<N: Scalar + AdditiveSemigroup, A: Space, B: Space> AbstractSemigroup<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}

impl<N: Scalar + AdditiveMonoid + Zero, A: Space, B: Space> AbstractMonoid<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}

impl<N: Scalar + ClosedNeg, A: Space, B: Space> TwoSidedInverse<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	fn two_sided_inverse(&self) -> Self {
		Self(self.0.two_sided_inverse())
	}
}

impl<N: Scalar + AdditiveQuasigroup + ClosedAdd + ClosedNeg, A: Space, B: Space> AbstractQuasigroup<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}

impl<N: Scalar + AdditiveLoop + ClosedAdd, A: Space, B: Space> AbstractLoop<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}

impl<N: Scalar + AdditiveGroup, A: Space, B: Space> AbstractGroup<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}

impl<N: Scalar + AdditiveGroup, A: Space, B: Space> AbstractGroupAbelian<Additive> for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{}






impl<N: Scalar + Zero + One, A: Space> Identity<Multiplicative> for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{
	fn identity() -> Self {
		Self(Identity::<Multiplicative>::identity())
	}
}

impl<N: Scalar + Zero + One + ClosedAdd + ClosedMul, A: Space> AbstractMagma<Multiplicative> for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{
	fn operate(&self, other: &Self) -> Self {
		Self(self.0.op(Multiplicative, &other.0))
	}
}

// Should require Rig, if it existed
impl<N: Scalar + Ring, A: Space> AbstractSemigroup<Multiplicative> for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{}

// Should require Rig, if it existed
impl<N: Scalar + Ring, A: Space> AbstractMonoid<Multiplicative> for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{}

impl<N: Scalar + Ring, A: Space> AbstractRing for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{}

impl<N: Scalar + RingCommutative, A: Space> AbstractRingCommutative for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{}

impl<N: Scalar + ComplexField, A: Space> TwoSidedInverse<Multiplicative> for Linear<N, A, A> where
	DefaultAllocator: Allocator<N, A::Dim, A::Dim>,
{
	fn two_sided_inverse(&self) -> Self {
		self.clone().inv()
	}
}


impl<N: Scalar + RingCommutative, A: Space, B: Space> AbstractModule for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type AbstractRing = N;
	fn multiply_by(&self, a: Self::AbstractRing) -> Self {
		Self(&self.0 * a)
	}
}

impl<N: Scalar + RingCommutative, A: Space, B: Space> Module for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Ring = N;
}

impl<N: Scalar + Field, A: Space, B: Space> VectorSpace for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
{
	type Field = N;
}

