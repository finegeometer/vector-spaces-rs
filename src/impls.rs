mod ops;
mod num_traits;
mod alga;

use crate::{Space, Linear};
use nalgebra::{Scalar, DefaultAllocator, allocator::Allocator};


impl<N: Scalar, A: Space, B: Space> Copy for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
	nalgebra::MatrixMN<N, B::Dim, A::Dim>: Copy,
{}

impl<N: Scalar, A: Space, B: Space> Clone for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>
{
	fn clone(&self) -> Self {
		Self(self.0.clone())
	}
}

impl<N: Scalar, A: Space, B: Space> core::fmt::Debug for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>
{
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		self.0.fmt(f)
	}
}

impl<N: Scalar, A: Space, B: Space> core::fmt::Display for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>,
	nalgebra::MatrixMN<N, B::Dim, A::Dim>: core::fmt::Display,
{
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		self.0.fmt(f)
	}
}

impl<N: Scalar, A: Space, B: Space> PartialEq for Linear<N, A, B> where
	DefaultAllocator: Allocator<N, B::Dim, A::Dim>
{
	fn eq(&self, other: &Self) -> bool {
		self == other
	}
}

