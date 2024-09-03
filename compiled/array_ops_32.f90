module array_ops_32
	use iso_c_binding
	implicit none

contains

! real: data type
! c_float: data precision
! C: lapack dtype char
! L, L0: eigpow limits
! ONE, ZERO: gemm constants
! SY: matrix form

subroutine matmul_multi_sym(A, b, npix, n, k) bind(c)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	integer(c_int), value         :: npix, n, k
	real(c_float),        intent(in)    :: A(n,n,npix)
	real(c_float),        intent(inout) :: b(n,k,npix)
	real(c_float)    :: x(size(b,1),size(b,2))
	integer(4) :: i
	!$omp parallel do private(i,x)
	do i = 1, size(A,3)
		x = b(:,:,i)
		b(:,:,i) = matmul(A(:,:,i),x)
	end do
end subroutine

subroutine matmul_multi(A, b, x, npix, n, m, k) bind(c)
	! This function assumes very small matrices, so it uses matmul instead of sgemm
	integer(c_int), value         :: npix, n, m, k
	real(c_float),        intent(in)    :: A(m,n,npix)
	real(c_float),        intent(in)    :: b(m,k,npix)
	real(c_float),        intent(inout) :: x(n,k,npix)
	integer(4) :: i
	!$omp parallel do private(i)
	do i = 1, size(A,3)
		x(:,:,i) = matmul(transpose(A(:,:,i)),b(:,:,i))
	end do
end subroutine

subroutine ang2rect(ang, rect, n) bind(c)
	integer(c_int), value      :: n
	real(c_float), intent(in)        :: ang(2,n)
	real(c_float), intent(inout)     :: rect(3,n)
	real(c_float) :: st,ct,sp,cp
	integer :: i
	!$omp parallel do private(i,st,ct,sp,cp)
	do i = 1, size(ang,2)
		sp = sin(ang(1,i)); cp = cos(ang(1,i))
		st = sin(ang(2,i)); ct = cos(ang(2,i))
		rect(1,i) = cp*ct
		rect(2,i) = sp*ct
		rect(3,i) = st
	end do
end subroutine

! Find areas in imap where values cross from below to above each
! value in vals, which must be sorted in ascending order. omap
! will be 0 in pixels where no crossing happens, and i where
! a crossing for vals(i) happens.
subroutine find_contours(imap, vals, omap, ny, nx, nv) bind(c)
	integer(c_int), value         :: ny, nx, nv
	real(c_float),        intent(in)    :: imap(nx,ny), vals(nv)
	integer(c_int), intent(inout) :: omap(nx,ny)
	integer, allocatable   :: work(:,:)
	real(c_float) :: v
	integer :: y, x, ip, i
	logical :: left, right
	allocate(work(ny,nx))
	do x = 1, nx
		do y = 1, ny
			ip = 1
			! Find which "bin" each value belongs in: 0 for for less
			! than vals(1), and so on
			v = imap(y,x)
			! nan is a pretty common case
			if(.not. (v .eq. v)) then
				work(y,x) = 1
				cycle
			end if
			left  = .true.
			right = .true.
			if(ip >   1) left  = v >= vals(ip-1)
			if(ip <= nv) right = v <  vals(ip)
			if(left .and. right) then
				i = ip
			else
				! Full search. No binary for now.
				do i = 1, nv
					if(v < vals(i)) exit
				end do
			end if
			work(y,x) = i
			ip = i
		end do
	end do
	! Edge detection
	omap = 0
	do x = 1, nx-1
		do y = 1, ny-1
			if(work(y,x) .ne. work(y+1,x)) then
				omap(y,x) = min(work(y,x),work(y+1,x))
			elseif(work(y,x) .ne. work(y,x+1)) then
				omap(y,x) = min(work(y,x),work(y,x+1))
			end if
		end do
	end do
end subroutine

subroutine roll_rows(imap, offsets, omap, ny, nx) bind(c)
	integer(c_int), value         :: ny, nx
	real(c_float),        intent(in)    :: imap(nx,ny)
	integer(c_int), intent(in)    :: offsets(ny)
	real(c_float),        intent(inout) :: omap(nx,ny)
	integer :: y, ix, ox
	!$omp parallel do private(ix,ox)
	do y = 1, ny
		do ix = 1, nx
			ox = ix + offsets(y)
			if    (ox < 1 ) then
				ox = ox + nx
			elseif(ox > nx) then
				ox = ox - nx
			end if
			omap(ox,y) = imap(ix,y)
		end do
	end do
end subroutine


end module
