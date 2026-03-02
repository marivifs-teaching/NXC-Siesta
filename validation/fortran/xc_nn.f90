! xc_nn.f90
module xc_nn
  implicit none
  private
  public :: load_xc_nn_weights
  public :: fx_forward_and_grad3, fc_forward_and_grad3
  public :: model_fx_from_rho_grad, model_fc_from_rho_grad
  public :: gelu, dgelu
  public :: nnxc_exc_spin
  public :: nnxc_edens_vxc_spin
  public :: nnxc_edens_vxc_spin_analytic

  integer, parameter :: dp = kind(1.0d0)
  integer, save :: debug_count_fc = 0

  ! Network architecture (can be overridden via the .fdf input).
  ! We compile with MAX sizes and use FX_IN/FX_H/FC_IN/FC_H at runtime.
  integer, parameter :: FX_IN_MAX = 3, FX_H_MAX = 256, FX_OUT = 1
  integer, parameter :: FC_IN_MAX = 3, FC_H_MAX = 256, FC_OUT = 1

  ! Defaults (match the common notebook/exporter setup):
  !   Fx: in=1 (uses x2 only), hidden=16
  !   Fc: in=3 (uses x0,x1,x2), hidden=16
  integer, save :: FX_IN = 1, FX_H = 16
  integer, save :: FC_IN = 3, FC_H = 16

  ! Weights and biases for Fx: shapes match numpy (out, in)

  ! Default prefix for NN weights location
  character(len=*), parameter :: prefix = "nn_params/xc_nn"
  real(dp), save :: W_fx1(FX_H_MAX, FX_IN_MAX), b_fx1(FX_H_MAX)
  real(dp), save :: W_fx2(FX_H_MAX, FX_H_MAX),  b_fx2(FX_H_MAX)
  real(dp), save :: W_fx3(FX_H_MAX, FX_H_MAX),  b_fx3(FX_H_MAX)
  real(dp), save :: W_fx4(FX_OUT,  FX_H_MAX),   b_fx4(FX_OUT)

  ! Weights and biases for Fc
  real(dp), save :: W_fc1(FC_H_MAX, FC_IN_MAX), b_fc1(FC_H_MAX)
  real(dp), save :: W_fc2(FC_H_MAX, FC_H_MAX),  b_fc2(FC_H_MAX)
  real(dp), save :: W_fc3(FC_H_MAX, FC_H_MAX),  b_fc3(FC_H_MAX)
  real(dp), save :: W_fc4(FC_OUT,  FC_H_MAX),   b_fc4(FC_OUT)

  ! Lieb–Oxford limits (must match your Python models)
  real(dp), parameter :: LOB_FX = 1.804_dp
  real(dp), parameter :: LOB_FC = 2.0_dp

contains

  !---------------- GELU and its derivative ----------------

  pure function gelu(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    real(dp), parameter :: sqrt2 = 1.4142135623730950488_dp
    real(dp) :: z
    z = x / sqrt2
    y = 0.5_dp * x * (1.0_dp + erf(z))
  end function gelu

  pure function dgelu(x) result(dy)
    real(dp), intent(in) :: x
    real(dp) :: dy
    real(dp), parameter :: sqrt2pi = 2.5066282746310005024_dp ! sqrt(2*pi)
    real(dp) :: z, expterm
    z = x / 1.4142135623730950488_dp
    expterm = exp(-0.5_dp * x * x)
    dy = 0.5_dp * (1.0_dp + erf(z)) + x * expterm / sqrt2pi
  end function dgelu

  !---------------- LOB and its derivative -----------------

  pure function lob(x, limit) result(y)
    real(dp), intent(in) :: x, limit
    real(dp) :: y
    real(dp) :: u, s
    u = x - log(limit - 1.0_dp)
    s = 1.0_dp / (1.0_dp + exp(-u))   ! sigmoid(u)
    y = limit * s - 1.0_dp
  end function lob

  pure function dlob(x, limit) result(dy)
    real(dp), intent(in) :: x, limit
    real(dp) :: dy
    real(dp) :: u, s
    u = x - log(limit - 1.0_dp)
    s = 1.0_dp / (1.0_dp + exp(-u))
    dy = limit * s * (1.0_dp - s)
  end function dlob

  !---------------- I/O helper -----------------

  subroutine read_matrix(filename, A)
    character(len=*), intent(in) :: filename
    real(dp), intent(out) :: A(:,:)
    integer :: iostat, i, j
    open(unit=77, file=filename, status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
       write(*,*) "Error opening ", trim(filename), " iostat=", iostat
       stop
    end if
    ! Read matrix in row-major order to match numpy's default
    read(77, *, iostat=iostat) ((A(i,j), j=1,size(A,2)), i=1,size(A,1))
    if (iostat /= 0) then
       write(*,*) "Error reading ", trim(filename), " iostat=", iostat, &
                  " expected shape=(", size(A,1), ",", size(A,2), ")"
       stop
    end if
    close(77)
  end subroutine read_matrix

  subroutine read_vector(filename, v)
    character(len=*), intent(in) :: filename
    real(dp), intent(out) :: v(:)
    integer :: iostat
    open(unit=78, file=filename, status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
       write(*,*) "Error opening ", trim(filename), " iostat=", iostat
       stop
    end if
    read(78, *, iostat=iostat) v
    if (iostat /= 0) then
       write(*,*) "Error reading ", trim(filename), " iostat=", iostat, &
                  " expected length=", size(v)
       stop
    end if
    close(78)
  end subroutine read_vector

  !---------------- Public loader -----------------

  subroutine load_xc_nn_weights(prefix)
    ! prefix is the same --out-prefix used in the exporter (e.g. 'xc_nn')
    character(len=*), intent(in) :: prefix
    character(len=256) :: fname

    write(*,*) "NNXC architecture (standalone): FX_IN=", FX_IN, " FX_H=", FX_H, " FC_IN=", FC_IN, " FC_H=", FC_H

    ! Fx
    write(fname,'(a,"_fx_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fx1(1:FX_H, 1:FX_IN))
    write(fname,'(a,"_fx_b1.dat")') trim(prefix)
    call read_vector(fname, b_fx1(1:FX_H))

    write(fname,'(a,"_fx_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fx2(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b2.dat")') trim(prefix)
    call read_vector(fname, b_fx2(1:FX_H))

    write(fname,'(a,"_fx_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fx3(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b3.dat")') trim(prefix)
    call read_vector(fname, b_fx3(1:FX_H))

    write(fname,'(a,"_fx_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fx4(1:FX_OUT, 1:FX_H))
    write(fname,'(a,"_fx_b4.dat")') trim(prefix)
    call read_vector(fname, b_fx4)

    ! Fc
    write(fname,'(a,"_fc_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fc1(1:FC_H, 1:FC_IN))
    write(fname,'(a,"_fc_b1.dat")') trim(prefix)
    call read_vector(fname, b_fc1(1:FC_H))

    write(fname,'(a,"_fc_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fc2(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b2.dat")') trim(prefix)
    call read_vector(fname, b_fc2(1:FC_H))

    write(fname,'(a,"_fc_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fc3(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b3.dat")') trim(prefix)
    call read_vector(fname, b_fc3(1:FC_H))

    write(fname,'(a,"_fc_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fc4(1:FC_OUT, 1:FC_H))
    write(fname,'(a,"_fc_b4.dat")') trim(prefix)
    call read_vector(fname, b_fc4)

    write(*,*) "Loaded XC NN weights with prefix ", trim(prefix)
  end subroutine load_xc_nn_weights

  !---------------- Fx MLP + LOB: forward + d/dx0,d/dx1,d/dx2 -----------------

  subroutine fx_forward_and_grad3(x0, x1, x2, Fx, dFx_dx0, dFx_dx1, dFx_dx2)
    ! Inputs:  x0, x1, x2
    ! Output:  Fx = 1 + LOB(x2 * MLP_x(x0,x1,x2))
    !          dFx_dx0, dFx_dx1, dFx_dx2
    real(dp), intent(in)  :: x0, x1, x2
    real(dp), intent(out) :: Fx, dFx_dx0, dFx_dx1, dFx_dx2

    real(dp) :: a0(FX_IN_MAX)
    real(dp) :: z1(FX_H_MAX), a1(FX_H_MAX)
    real(dp) :: z2(FX_H_MAX), a2(FX_H_MAX)
    real(dp) :: z3(FX_H_MAX), a3(FX_H_MAX)
    real(dp) :: z4(FX_OUT)
    real(dp) :: y
    real(dp) :: delta3(FX_H_MAX), delta2(FX_H_MAX), delta1(FX_H_MAX)
    real(dp) :: grad_in(FX_IN_MAX)
    real(dp) :: zlob, dlobdz
    integer :: i, j

    ! Forward: map (x0,x1,x2) -> NN inputs
    a0 = 0.0_dp
    if (FX_IN == 1) then
       a0(1) = x2
    else if (FX_IN == 3) then
       a0(1) = x0
       a0(2) = x1
       a0(3) = x2
    else
       write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
       stop
    end if

    do i = 1, FX_H
       z1(i) = b_fx1(i)
       do j = 1, FX_IN
          z1(i) = z1(i) + W_fx1(i,j) * a0(j)
       end do
       a1(i) = gelu(z1(i))
    end do

    do i = 1, FX_H
       z2(i) = b_fx2(i)
       do j = 1, FX_H
          z2(i) = z2(i) + W_fx2(i,j) * a1(j)
       end do
       a2(i) = gelu(z2(i))
    end do

    do i = 1, FX_H
       z3(i) = b_fx3(i)
       do j = 1, FX_H
          z3(i) = z3(i) + W_fx3(i,j) * a2(j)
       end do
       a3(i) = gelu(z3(i))
    end do

    do i = 1, FX_OUT
       z4(i) = b_fx4(i)
       do j = 1, FX_H
          z4(i) = z4(i) + W_fx4(i,j) * a3(j)
       end do
    end do
    y = z4(1)

    ! Apply LOB: Fx = 1 + lob(x2 * y, LOB_FX)
    zlob = x2 * y
    Fx   = 1.0_dp + lob(zlob, LOB_FX)

    ! Backprop to get dy/dx0,dy/dx1,dy/dx2
    do i = 1, FX_H
       delta3(i) = 0.0_dp
       do j = 1, FX_OUT
          delta3(i) = delta3(i) + W_fx4(j,i) * 1.0_dp
       end do
       delta3(i) = delta3(i) * dgelu(z3(i))
    end do

    do i = 1, FX_H
       delta2(i) = 0.0_dp
       do j = 1, FX_H
          delta2(i) = delta2(i) + W_fx3(j,i) * delta3(j)
       end do
       delta2(i) = delta2(i) * dgelu(z2(i))
    end do

    do i = 1, FX_H
       delta1(i) = 0.0_dp
       do j = 1, FX_H
          delta1(i) = delta1(i) + W_fx2(j,i) * delta2(j)
       end do
       delta1(i) = delta1(i) * dgelu(z1(i))
    end do

    grad_in = 0.0_dp
    do j = 1, FX_IN
       do i = 1, FX_H
          grad_in(j) = grad_in(j) + W_fx1(i,j) * delta1(i)
       end do
    end do

    dlobdz  = dlob(zlob, LOB_FX)
    ! Fx = 1 + LOB(x2 * y)
    if (FX_IN == 1) then
       dFx_dx0 = 0.0_dp
       dFx_dx1 = 0.0_dp
       dFx_dx2 = dlobdz * ( y  + x2 * grad_in(1) )
    else if (FX_IN == 3) then
       dFx_dx0 = dlobdz * ( x2 * grad_in(1) )
       dFx_dx1 = dlobdz * ( x2 * grad_in(2) )
       dFx_dx2 = dlobdz * ( y  + x2 * grad_in(3) )
    else
       write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
       stop
    end if

  end subroutine fx_forward_and_grad3

  !---------------- Fc MLP + LOB: forward + d/dx0,d/dx1,d/dx2 -----------------

  subroutine fc_forward_and_grad3(x0, x1, x2, Fc, dFc_dx0, dFc_dx1, dFc_dx2)
    ! Inputs:  x0, x1, x2
    ! Output:  Fc = 1 + LOB(x2 * MLP_c(x0,x1,x2))
    !          dFc_dx0, dFc_dx1, dFc_dx2
    real(dp), intent(in)  :: x0, x1, x2
    real(dp), intent(out) :: Fc, dFc_dx0, dFc_dx1, dFc_dx2

    real(dp) :: a0(FC_IN_MAX)
    real(dp) :: z1(FC_H_MAX), a1(FC_H_MAX)
    real(dp) :: z2(FC_H_MAX), a2(FC_H_MAX)
    real(dp) :: z3(FC_H_MAX), a3(FC_H_MAX)
    real(dp) :: z4(FC_OUT)
    real(dp) :: y
    real(dp) :: delta3(FC_H_MAX), delta2(FC_H_MAX), delta1(FC_H_MAX)
    real(dp) :: grad_in(FC_IN_MAX)
    real(dp) :: zlob, dlobdz
    integer :: i, j

    ! Forward: map (x0,x1,x2) -> NN inputs
    a0 = 0.0_dp
    if (FC_IN == 1) then
       a0(1) = x2
    else if (FC_IN == 3) then
       a0(1) = x0
       a0(2) = x1
       a0(3) = x2
    else
       write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
       stop
    end if

    do i = 1, FC_H
       z1(i) = b_fc1(i)
       do j = 1, FC_IN
          z1(i) = z1(i) + W_fc1(i,j) * a0(j)
       end do
       a1(i) = gelu(z1(i))
    end do

    do i = 1, FC_H
       z2(i) = b_fc2(i)
       do j = 1, FC_H
          z2(i) = z2(i) + W_fc2(i,j) * a1(j)
       end do
       a2(i) = gelu(z2(i))
    end do

    do i = 1, FC_H
       z3(i) = b_fc3(i)
       do j = 1, FC_H
          z3(i) = z3(i) + W_fc3(i,j) * a2(j)
       end do
       a3(i) = gelu(z3(i))
    end do

    do i = 1, FC_OUT
       z4(i) = b_fc4(i)
       do j = 1, FC_H
          z4(i) = z4(i) + W_fc4(i,j) * a3(j)
       end do
    end do
    y = z4(1)

    zlob = x2 * y
    Fc   = 1.0_dp + lob(zlob, LOB_FC)

    ! Backprop to get dy/dx0, dy/dx1, dy/dx2
    do i = 1, FC_H
       delta3(i) = 0.0_dp
       do j = 1, FC_OUT
          delta3(i) = delta3(i) + W_fc4(j,i) * 1.0_dp
       end do
       delta3(i) = delta3(i) * dgelu(z3(i))
    end do

    do i = 1, FC_H
       delta2(i) = 0.0_dp
       do j = 1, FC_H
          delta2(i) = delta2(i) + W_fc3(j,i) * delta3(j)
       end do
       delta2(i) = delta2(i) * dgelu(z2(i))
    end do

    do i = 1, FC_H
       delta1(i) = 0.0_dp
       do j = 1, FC_H
          delta1(i) = delta1(i) + W_fc2(j,i) * delta2(j)
       end do
       delta1(i) = delta1(i) * dgelu(z1(i))
    end do

    grad_in = 0.0_dp
    do j = 1, FC_IN
       do i = 1, FC_H
          grad_in(j) = grad_in(j) + W_fc1(i,j) * delta1(i)
       end do
    end do

    dlobdz   = dlob(zlob, LOB_FC)
    ! Fc = 1 + LOB(x2 * y)
    if (FC_IN == 1) then
       dFc_dx0 = 0.0_dp
       dFc_dx1 = 0.0_dp
       dFc_dx2 = dlobdz * ( y  + x2 * grad_in(1) )
    else if (FC_IN == 3) then
       dFc_dx0 = dlobdz * ( x2 * grad_in(1) )
       dFc_dx1 = dlobdz * ( x2 * grad_in(2) )
       dFc_dx2 = dlobdz * ( y  + x2 * grad_in(3) )
    else
       write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
       stop
    end if

    ! Debug print disabled (kept counter for optional future use)
    debug_count_fc = debug_count_fc + 1

  end subroutine fc_forward_and_grad3


  !---------------- Top-level model wrappers: (rho,grad_rho) -> Fx,Fc -----------------

  subroutine model_fx_from_rho_grad(rho, grad_rho, Fx)
    !! Fortran version of GGA_Fx_G_transf_lin.__call__(inputs=[rho, grad_rho])
    implicit none
    real(dp), intent(in)  :: rho, grad_rho
    real(dp), intent(out) :: Fx
    real(dp) :: kF, s, s2, x0, x1, x2
    real(dp) :: dFx_dx0, dFx_dx1, dFx_dx2
    real(dp), parameter :: pi = acos(-1.0_dp)

    if (rho <= 0.0_dp) then
       Fx = 1.0_dp
       return
    end if

    ! k_F = (3 * pi^2 * rho)^(1/3)
    kF = (3.0_dp * pi*pi * rho)**(1.0_dp/3.0_dp)
    ! s = |grad rho| / (2 k_F rho)
    s  = grad_rho / (2.0_dp * kF * rho)
    s2 = s*s

    ! x2 = log(1 + s) * (1 - exp(-s^2))
    x2 = log(1.0_dp + s) * (1.0_dp - exp(-s2))

    ! x0 = log(rho^(1/3) + 1.0d-5)
    x0 = log(rho**(1.0_dp/3.0_dp) + 1.0d-5)

    ! Unpolarized spin feature: zeta' = 1 => x1 = log(1 + 1.0d-5)
    x1 = log(1.0_dp + 1.0d-5)

    call fx_forward_and_grad3(x0, x1, x2, Fx, dFx_dx0, dFx_dx1, dFx_dx2)
  end subroutine model_fx_from_rho_grad


  subroutine model_fc_from_rho_grad(rho, grad_rho, Fc)
    !! Fortran version of GGA_Fc_G_transf_lin.__call__(inputs=[rho, grad_rho])
    implicit none
    real(dp), intent(in)  :: rho, grad_rho
    real(dp), intent(out) :: Fc
    real(dp) :: kF, s, s2, x0, x1, x2
    real(dp) :: dFc_dx0, dFc_dx1, dFc_dx2
    real(dp), parameter :: eps = 1.0d-5
    real(dp), parameter :: pi  = acos(-1.0_dp)

    if (rho <= 0.0_dp) then
       Fc = 1.0_dp
       return
    end if

    ! k_F = (3 * pi^2 * rho)^(1/3)
    kF = (3.0_dp * pi*pi * rho)**(1.0_dp/3.0_dp)
    ! s = |grad rho| / (2 k_F rho)
    s  = grad_rho / (2.0_dp * kF * rho)
    s2 = s*s

    ! x2 = log(1 + s) * (1 - exp(-s^2))
    x2 = log(1.0_dp + s) * (1.0_dp - exp(-s2))

    ! x0 = log(rho^(1/3) + eps)
    x0 = log(rho**(1.0_dp/3.0_dp) + eps)

    ! Unpolarized spin feature: zeta' = 1 => x1 = log(1 + eps)
    x1 = log(1.0_dp + eps)

    call fc_forward_and_grad3(x0, x1, x2, Fc, dFc_dx0, dFc_dx1, dFc_dx2)
  end subroutine model_fc_from_rho_grad


  !---------------- PW92 correlation (spin-interpolated) -----------------

  pure function pw92_eps_c_spin(rho, zeta) result(epsc)
    ! PW92 correlation energy per electron with spin interpolation.
    ! Matches the notebook's __pw92_eps_c_wospin_point.
    real(dp), intent(in) :: rho, zeta
    real(dp) :: epsc

    real(dp), parameter :: A(3)      = (/ 0.031090690869654895_dp, 0.015545_dp, 0.016887_dp /)
    real(dp), parameter :: ALPHA1(3) = (/ 0.21370_dp, 0.20548_dp, 0.11125_dp /)
    real(dp), parameter :: BETA1(3)  = (/ 7.5957_dp, 14.1189_dp, 10.357_dp /)
    real(dp), parameter :: BETA2(3)  = (/ 3.5876_dp, 6.1977_dp, 3.6231_dp /)
    real(dp), parameter :: BETA3(3)  = (/ 1.6382_dp, 3.3662_dp, 0.88026_dp /)
    real(dp), parameter :: BETA4(3)  = (/ 0.49294_dp, 0.62517_dp, 0.49671_dp /)

    real(dp) :: rs, B, C, G(3)
    real(dp) :: zc, C0, F, Fpp0, G0, G1, G2
    integer :: k

    if (rho <= 0.0_dp) then
       epsc = 0.0_dp
       return
    end if

    rs = (3.0_dp / (4.0_dp * acos(-1.0_dp) * rho))**(1.0_dp/3.0_dp)

    do k = 1, 3
       B = BETA1(k)*sqrt(rs) + BETA2(k)*rs + BETA3(k)*rs**1.5_dp + BETA4(k)*rs**2
       C = 1.0_dp + 1.0_dp / (2.0_dp * A(k) * B)
       G(k) = -2.0_dp * A(k) * (1.0_dp + ALPHA1(k)*rs) * log(C)
    end do

    ! Clamp zeta to avoid tiny numerical overshoots
    zc = max(-1.0_dp + 1.0d-12, min(1.0_dp - 1.0d-12, zeta))

    C0   = 1.0_dp / (2.0_dp**(4.0_dp/3.0_dp) - 2.0_dp)
    F    = ( (1.0_dp+zc)**(4.0_dp/3.0_dp) + (1.0_dp-zc)**(4.0_dp/3.0_dp) - 2.0_dp ) * C0
    Fpp0 = C0 * 8.0_dp/9.0_dp

    G0 = G(1)
    G1 = G(2)
    G2 = G(3)

    epsc = G0 - G2 * F / Fpp0 * (1.0_dp - zc**4) + (G1 - G0) * F * zc**4
  end function pw92_eps_c_spin


  subroutine pw92_eps_c_spin_derivs(rho, zeta, deps_drho, deps_dz)
    ! Analytic partial derivatives of pw92_eps_c_spin(rho,zeta).
    real(dp), intent(in)  :: rho, zeta
    real(dp), intent(out) :: deps_drho, deps_dz

    real(dp), parameter :: A(3)      = (/ 0.031090690869654895_dp, 0.015545_dp, 0.016887_dp /)
    real(dp), parameter :: ALPHA1(3) = (/ 0.21370_dp, 0.20548_dp, 0.11125_dp /)
    real(dp), parameter :: BETA1(3)  = (/ 7.5957_dp, 14.1189_dp, 10.357_dp /)
    real(dp), parameter :: BETA2(3)  = (/ 3.5876_dp, 6.1977_dp, 3.6231_dp /)
    real(dp), parameter :: BETA3(3)  = (/ 1.6382_dp, 3.3662_dp, 0.88026_dp /)
    real(dp), parameter :: BETA4(3)  = (/ 0.49294_dp, 0.62517_dp, 0.49671_dp /)

    real(dp) :: rs, drs_drho
    real(dp) :: B(3), C(3), logC(3), dBdrs(3), dGdrs(3), G(3)
    real(dp) :: C0, F, Fpp0, dFdz
    real(dp) :: zc
    integer :: k

    if (rho <= 0.0_dp) then
       deps_drho = 0.0_dp
       deps_dz   = 0.0_dp
       return
    end if

    zc = max(-1.0_dp + 1.0d-12, min(1.0_dp - 1.0d-12, zeta))

    rs = (3.0_dp / (4.0_dp * acos(-1.0_dp) * rho))**(1.0_dp/3.0_dp)
    drs_drho = -(1.0_dp/3.0_dp) * rs / rho

    do k = 1, 3
       B(k) = BETA1(k)*sqrt(rs) + BETA2(k)*rs + BETA3(k)*rs**1.5_dp + BETA4(k)*rs**2
       dBdrs(k) = BETA1(k)*(0.5_dp/sqrt(rs)) + BETA2(k) + BETA3(k)*1.5_dp*sqrt(rs) + BETA4(k)*2.0_dp*rs
       C(k) = 1.0_dp + 1.0_dp/(2.0_dp*A(k)*B(k))
       logC(k) = log(C(k))
       G(k) = -2.0_dp*A(k)*(1.0_dp + ALPHA1(k)*rs)*logC(k)
       dGdrs(k) = -2.0_dp*A(k)*ALPHA1(k)*logC(k) + (1.0_dp + ALPHA1(k)*rs) * dBdrs(k) / (B(k)*B(k)*C(k))
    end do

    C0   = 1.0_dp / (2.0_dp**(4.0_dp/3.0_dp) - 2.0_dp)
    F    = ( (1.0_dp+zc)**(4.0_dp/3.0_dp) + (1.0_dp-zc)**(4.0_dp/3.0_dp) - 2.0_dp ) * C0
    Fpp0 = C0 * 8.0_dp/9.0_dp
    dFdz = C0 * (4.0_dp/3.0_dp) * ( (1.0_dp+zc)**(1.0_dp/3.0_dp) - (1.0_dp-zc)**(1.0_dp/3.0_dp) )

    ! deps/dzeta
    deps_dz = (-G(3)/Fpp0) * ( dFdz*(1.0_dp - zc**4) + F*(-4.0_dp*zc**3) ) + (G(2)-G(1)) * ( dFdz*zc**4 + F*(4.0_dp*zc**3) )

    ! deps/drho via rs
    deps_drho = ( dGdrs(1) - dGdrs(3)*F/Fpp0*(1.0_dp - zc**4) + (dGdrs(2) - dGdrs(1))*F*zc**4 ) * drs_drho

  end subroutine pw92_eps_c_spin_derivs


  !---------------- Transformed inputs (match notebook) -----------------

  pure subroutine transform_inputs(rho_tot, zeta, grad_mod, x0, x1, x2)
    ! x0 = log(rho^{1/3} + eps)
    ! x1 = log(zeta' + eps), zeta' = 1/2*((1+zeta)^(4/3) + (1-zeta)^(4/3))
    ! x2 = (1-exp(-s^2))*log(1+s),  s = |grad rho|/(2 kF rho)
    real(dp), intent(in)  :: rho_tot, zeta, grad_mod
    real(dp), intent(out) :: x0, x1, x2

    real(dp), parameter :: eps = 1.0d-5
    real(dp), parameter :: pi  = acos(-1.0_dp)
    real(dp) :: kF, s, s2, zc, zeta_p

    if (rho_tot <= 0.0_dp) then
       x0 = 0.0_dp
       x1 = 0.0_dp
       x2 = 0.0_dp
       return
    end if

    zc = max(-1.0_dp + 1.0d-12, min(1.0_dp - 1.0d-12, zeta))

    x0 = log(rho_tot**(1.0_dp/3.0_dp) + eps)

    zeta_p = 0.5_dp * ( (1.0_dp+zc)**(4.0_dp/3.0_dp) + (1.0_dp-zc)**(4.0_dp/3.0_dp) )
    x1 = log(zeta_p + eps)

    kF = (3.0_dp * pi*pi * rho_tot)**(1.0_dp/3.0_dp)
    s  = grad_mod / (2.0_dp * kF * rho_tot)
    s2 = s*s
    x2 = (1.0_dp - exp(-s2)) * log(1.0_dp + s)
  end subroutine transform_inputs


  !---------------- Energy density per electron (spin) -----------------

  pure function epsx_lda_unpol(rho) result(epsx)
    ! Unpolarized LDA exchange energy per electron
    real(dp), intent(in) :: rho
    real(dp) :: epsx
    real(dp), parameter :: c = -0.75_dp * (3.0_dp/acos(-1.0_dp))**(1.0_dp/3.0_dp)
    if (rho <= 0.0_dp) then
       epsx = 0.0_dp
    else
       epsx = c * rho**(1.0_dp/3.0_dp)
    end if
  end function epsx_lda_unpol


  subroutine nnxc_exc_spin(rho_u, rho_d, gxu, gyu, gzu, gxd, gyd, gzd, exc)
    ! Return eps_xc (per electron) for spin-polarized inputs, matching the PySCF notebook.
    ! Grid inputs are densities and Cartesian gradients (atomic units).
    real(dp), intent(in)  :: rho_u, rho_d
    real(dp), intent(in)  :: gxu, gyu, gzu, gxd, gyd, gzd
    real(dp), intent(out) :: exc

    real(dp) :: rho_tot, zeta
    real(dp) :: gu_mod, gd_mod, gtot_mod
    real(dp) :: x0, x1, x2, x0c, x1c, x2c
    real(dp) :: Fx_u, Fx_d, Fc
    real(dp) :: dum0, dum1, dum2
    real(dp) :: ex_dens, ec_dens
    real(dp) :: epsc

    rho_tot = rho_u + rho_d
    if (rho_tot <= 0.0_dp) then
       exc = 0.0_dp
       return
    end if

    ! gradient magnitudes
    gu_mod   = sqrt(gxu*gxu + gyu*gyu + gzu*gzu)
    gd_mod   = sqrt(gxd*gxd + gyd*gyd + gzd*gzd)
    gtot_mod = sqrt((gxu+gxd)**2 + (gyu+gyd)**2 + (gzu+gzd)**2)

    !---------------- Exchange: spin-scaling ----------------
    ex_dens = 0.0_dp

    if (rho_u > 0.0_dp) then
       call transform_inputs(2.0_dp*rho_u, 0.0_dp, 2.0_dp*gu_mod, x0, x1, x2)
       call fx_forward_and_grad3(x0, x1, x2, Fx_u, dum0, dum1, dum2)
       ex_dens = ex_dens + rho_u * Fx_u * epsx_lda_unpol(2.0_dp*rho_u)
    end if

    if (rho_d > 0.0_dp) then
       call transform_inputs(2.0_dp*rho_d, 0.0_dp, 2.0_dp*gd_mod, x0, x1, x2)
       call fx_forward_and_grad3(x0, x1, x2, Fx_d, dum0, dum1, dum2)
       ex_dens = ex_dens + rho_d * Fx_d * epsx_lda_unpol(2.0_dp*rho_d)
    end if

    !---------------- Correlation: total rho, zeta, sigma_tot ----------------
    zeta = (rho_u - rho_d) / rho_tot
    call transform_inputs(rho_tot, zeta, gtot_mod, x0, x1, x2)
    call fc_forward_and_grad3(x0, x1, x2, Fc, dum0, dum1, dum2)

    epsc   = pw92_eps_c_spin(rho_tot, zeta)
    ec_dens = rho_tot * Fc * epsc

    exc = (ex_dens + ec_dens) / rho_tot
  end subroutine nnxc_exc_spin


  !---------------- Energy density + potentials (debug: finite-difference) -----------------

  function nnxc_edens_from_rhosigma(rho_u, rho_d, suu, sud, sdd) result(edens)
    ! Returns energy density e_xc = rho_tot * eps_xc, using the same functional
    ! definition as nnxc_exc_spin, but taking scalar sigma components.
    real(dp), intent(in) :: rho_u, rho_d, suu, sud, sdd
    real(dp) :: edens

    real(dp) :: rho_tot, zeta
    real(dp) :: gu_mod, gd_mod, gtot_mod
    real(dp) :: x0, x1, x2
    real(dp) :: Fx_u, Fx_d, Fc
    real(dp) :: dum0, dum1, dum2
    real(dp) :: ex_dens, ec_dens
    real(dp) :: epsc

    rho_tot = rho_u + rho_d
    if (rho_tot <= 0.0_dp) then
       edens = 0.0_dp
       return
    end if

    gu_mod   = sqrt(max(0.0_dp, suu))
    gd_mod   = sqrt(max(0.0_dp, sdd))
    gtot_mod = sqrt(max(0.0_dp, suu + 2.0_dp*sud + sdd))

    ! Exchange: spin-scaling
    ex_dens = 0.0_dp
    if (rho_u > 0.0_dp) then
       call transform_inputs(2.0_dp*rho_u, 0.0_dp, 2.0_dp*gu_mod, x0, x1, x2)
       call fx_forward_and_grad3(x0, x1, x2, Fx_u, dum0, dum1, dum2)
       ex_dens = ex_dens + rho_u * Fx_u * epsx_lda_unpol(2.0_dp*rho_u)
    end if
    if (rho_d > 0.0_dp) then
       call transform_inputs(2.0_dp*rho_d, 0.0_dp, 2.0_dp*gd_mod, x0, x1, x2)
       call fx_forward_and_grad3(x0, x1, x2, Fx_d, dum0, dum1, dum2)
       ex_dens = ex_dens + rho_d * Fx_d * epsx_lda_unpol(2.0_dp*rho_d)
    end if

    ! Correlation: total rho, zeta, sigma_tot
    zeta = (rho_u - rho_d) / rho_tot
    call transform_inputs(rho_tot, zeta, gtot_mod, x0, x1, x2)
    call fc_forward_and_grad3(x0, x1, x2, Fc, dum0, dum1, dum2)

    epsc     = pw92_eps_c_spin(rho_tot, zeta)
    ec_dens  = rho_tot * Fc * epsc

    edens = ex_dens + ec_dens
  end function nnxc_edens_from_rhosigma


  subroutine nnxc_edens_vxc_spin_analytic(rho_u, rho_d, suu, sud, sdd, edens, vrhou, vrhod, vsuu, vsud, vsdd)
    ! Analytic PySCF-style potentials for e_xc = rho_tot * eps_xc.
    real(dp), intent(in)  :: rho_u, rho_d, suu, sud, sdd
    real(dp), intent(out) :: edens, vrhou, vrhod, vsuu, vsud, vsdd

    real(dp), parameter :: eps = 1.0d-5
    real(dp), parameter :: pi  = acos(-1.0_dp)
    real(dp), parameter :: tiny = 1.0d-14
    real(dp) :: rho_tot, zeta, zc
    real(dp) :: sigtot, gu, gd, gtot

    ! NN / features
    real(dp) :: x0, x1, x2, x0c, x1c, x2c
    real(dp) :: Fx, dFx_x0, dFx_x1, dFx_x2
    real(dp) :: Fc, dFc_x0, dFc_x1, dFc_x2

    ! Exchange helpers
    real(dp) :: rho2, g2, kF, s, s2
    real(dp) :: epsx, depsx_drho2
    real(dp) :: dx0_drho2
    real(dp) :: ds_drho2, ds_dg2
    real(dp) :: f_s, g_s, fp_s, gp_s
    real(dp) :: dx2_ds, dx2_drho2, dx2_dg2
    real(dp) :: dFx_drho2, dFx_drho, dFx_dsigma

    ! Correlation helpers
    real(dp) :: kF_c, s_c
    real(dp) :: epsc, depsc_drho, depsc_dz
    real(dp) :: dx0c_drho, zeta_p, dzeta_p_dz, dx1c_dz
    real(dp) :: ds_c_drho, ds_c_dgtot
    real(dp) :: dx2c_ds, dx2c_drho, dx2c_dgtot
    real(dp) :: dFc_drho, dFc_dz, dFc_dsigtot
    real(dp) :: dec_drho_tot, dec_dz, dec_dsigtot
    real(dp) :: dz_drhou, dz_drhod

    ! accumulators
    real(dp) :: dex_drhou, dex_drhod, dex_dsuu, dex_dsdd

    rho_tot = rho_u + rho_d
    if (rho_tot <= 0.0_dp) then
       edens = 0.0_dp
       vrhou = 0.0_dp
       vrhod = 0.0_dp
       vsuu  = 0.0_dp
       vsud  = 0.0_dp
       vsdd  = 0.0_dp
       return
    end if

    sigtot = suu + 2.0_dp*sud + sdd
    if (sigtot < 0.0_dp) sigtot = 0.0_dp

    gu   = sqrt(max(0.0_dp, suu))
    gd   = sqrt(max(0.0_dp, sdd))
    gtot = sqrt(sigtot)

    ! start accumulators
    dex_drhou = 0.0_dp
    dex_drhod = 0.0_dp
    dex_dsuu  = 0.0_dp
    dex_dsdd  = 0.0_dp

    ! ===================== Exchange (spin scaling) =====================
    ! UP channel
    if (rho_u > tiny) then
       rho2 = 2.0_dp * rho_u
       g2   = 2.0_dp * gu

       x0 = log(rho2**(1.0_dp/3.0_dp) + eps)
       x1 = log(1.0_dp + eps)

       kF = (3.0_dp*pi*pi*rho2)**(1.0_dp/3.0_dp)
       s  = g2 / (2.0_dp*kF*rho2)
       s2 = s*s
       x2 = (1.0_dp - exp(-s2)) * log(1.0_dp + s)

       call fx_forward_and_grad3(x0, x1, x2, Fx, dFx_x0, dFx_x1, dFx_x2)

       epsx = epsx_lda_unpol(rho2)
       depsx_drho2 = (-0.75_dp*(3.0_dp/pi)**(1.0_dp/3.0_dp))*(1.0_dp/3.0_dp)*rho2**(-2.0_dp/3.0_dp)

       dx0_drho2 = ( (1.0_dp/3.0_dp)*rho2**(-2.0_dp/3.0_dp) ) / (rho2**(1.0_dp/3.0_dp) + eps)
       ds_drho2  = -(4.0_dp/3.0_dp) * s / rho2
       ds_dg2    = 1.0_dp / (2.0_dp*kF*rho2)

       f_s  = 1.0_dp - exp(-s2)
       g_s  = log(1.0_dp + s)
       fp_s = 2.0_dp*s*exp(-s2)
       gp_s = 1.0_dp/(1.0_dp + s)
       dx2_ds = fp_s*g_s + f_s*gp_s

       dx2_drho2 = dx2_ds * ds_drho2
       dx2_dg2   = dx2_ds * ds_dg2

       dFx_drho2 = dFx_x0*dx0_drho2 + dFx_x2*dx2_drho2
       dFx_drho  = 2.0_dp * dFx_drho2

       if (suu > 0.0_dp) then
          dFx_dsigma = dFx_x2*dx2_dg2 * (1.0_dp/sqrt(suu))
       else
          dFx_dsigma = 0.0_dp
       end if

       dex_drhou = dex_drhou + Fx*epsx + rho_u*( dFx_drho*epsx + Fx*(2.0_dp*depsx_drho2) )
       dex_dsuu  = dex_dsuu  + rho_u*epsx*dFx_dsigma
    end if

    ! DOWN channel
    if (rho_d > tiny) then
       rho2 = 2.0_dp * rho_d
       g2   = 2.0_dp * gd

       x0 = log(rho2**(1.0_dp/3.0_dp) + eps)
       x1 = log(1.0_dp + eps)

       kF = (3.0_dp*pi*pi*rho2)**(1.0_dp/3.0_dp)
       s  = g2 / (2.0_dp*kF*rho2)
       s2 = s*s
       x2 = (1.0_dp - exp(-s2)) * log(1.0_dp + s)

       call fx_forward_and_grad3(x0, x1, x2, Fx, dFx_x0, dFx_x1, dFx_x2)

       epsx = epsx_lda_unpol(rho2)
       depsx_drho2 = (-0.75_dp*(3.0_dp/pi)**(1.0_dp/3.0_dp))*(1.0_dp/3.0_dp)*rho2**(-2.0_dp/3.0_dp)

       dx0_drho2 = ( (1.0_dp/3.0_dp)*rho2**(-2.0_dp/3.0_dp) ) / (rho2**(1.0_dp/3.0_dp) + eps)
       ds_drho2  = -(4.0_dp/3.0_dp) * s / rho2
       ds_dg2    = 1.0_dp / (2.0_dp*kF*rho2)

       f_s  = 1.0_dp - exp(-s2)
       g_s  = log(1.0_dp + s)
       fp_s = 2.0_dp*s*exp(-s2)
       gp_s = 1.0_dp/(1.0_dp + s)
       dx2_ds = fp_s*g_s + f_s*gp_s

       dx2_drho2 = dx2_ds * ds_drho2
       dx2_dg2   = dx2_ds * ds_dg2

       dFx_drho2 = dFx_x0*dx0_drho2 + dFx_x2*dx2_drho2
       dFx_drho  = 2.0_dp * dFx_drho2

       if (sdd > 0.0_dp) then
          dFx_dsigma = dFx_x2*dx2_dg2 * (1.0_dp/sqrt(sdd))
       else
          dFx_dsigma = 0.0_dp
       end if

       dex_drhod = dex_drhod + Fx*epsx + rho_d*( dFx_drho*epsx + Fx*(2.0_dp*depsx_drho2) )
       dex_dsdd  = dex_dsdd  + rho_d*epsx*dFx_dsigma
    end if

    ! ===================== Correlation =====================
    zeta = (rho_u - rho_d) / rho_tot
    zc   = max(-1.0_dp + 1.0d-12, min(1.0_dp - 1.0d-12, zeta))

    x0c = log(rho_tot**(1.0_dp/3.0_dp) + eps)
    dx0c_drho = ( (1.0_dp/3.0_dp)*rho_tot**(-2.0_dp/3.0_dp) ) / (rho_tot**(1.0_dp/3.0_dp) + eps)

    zeta_p = 0.5_dp * ( (1.0_dp+zc)**(4.0_dp/3.0_dp) + (1.0_dp-zc)**(4.0_dp/3.0_dp) )
    dzeta_p_dz = (2.0_dp/3.0_dp) * ( (1.0_dp+zc)**(1.0_dp/3.0_dp) - (1.0_dp-zc)**(1.0_dp/3.0_dp) )
    dx1c_dz = dzeta_p_dz / (zeta_p + eps)
    x1c = log(zeta_p + eps)

    kF_c = (3.0_dp*pi*pi*rho_tot)**(1.0_dp/3.0_dp)
    s_c  = gtot / (2.0_dp*kF_c*rho_tot)
    x2c  = (1.0_dp - exp(-(s_c*s_c))) * log(1.0_dp + s_c)

    ds_c_drho  = -(4.0_dp/3.0_dp) * s_c / rho_tot
    ds_c_dgtot = 1.0_dp / (2.0_dp*kF_c*rho_tot)

    f_s  = 1.0_dp - exp(-(s_c*s_c))
    g_s  = log(1.0_dp + s_c)
    fp_s = 2.0_dp*s_c*exp(-(s_c*s_c))
    gp_s = 1.0_dp/(1.0_dp + s_c)
    dx2c_ds = fp_s*g_s + f_s*gp_s

    dx2c_drho  = dx2c_ds * ds_c_drho
    dx2c_dgtot = dx2c_ds * ds_c_dgtot

    call fc_forward_and_grad3(x0c, x1c, x2c, Fc, dFc_x0, dFc_x1, dFc_x2)

    epsc = pw92_eps_c_spin(rho_tot, zc)
    call pw92_eps_c_spin_derivs(rho_tot, zc, depsc_drho, depsc_dz)

    dFc_drho = dFc_x0*dx0c_drho + dFc_x2*dx2c_drho
    dFc_dz   = dFc_x1*dx1c_dz

    dec_drho_tot = Fc*epsc + rho_tot*( dFc_drho*epsc + Fc*depsc_drho )
    dec_dz       = rho_tot*( dFc_dz*epsc + Fc*depsc_dz )

    if (sigtot > tiny) then
       dFc_dsigtot = dFc_x2*dx2c_dgtot * (1.0_dp/(2.0_dp*gtot))
    else
       dFc_dsigtot = 0.0_dp
    end if
    dec_dsigtot = rho_tot*epsc*dFc_dsigtot

    dz_drhou = (2.0_dp*rho_d)/(rho_tot*rho_tot)
    dz_drhod = (-2.0_dp*rho_u)/(rho_tot*rho_tot)

    edens = nnxc_edens_from_rhosigma(rho_u, rho_d, suu, sud, sdd)

    vrhou = dex_drhou + dec_drho_tot + dec_dz*dz_drhou
    vrhod = dex_drhod + dec_drho_tot + dec_dz*dz_drhod

    vsuu = dex_dsuu + dec_dsigtot
    vsud = 2.0_dp*dec_dsigtot
    vsdd = dex_dsdd + dec_dsigtot

  end subroutine nnxc_edens_vxc_spin_analytic

  subroutine nnxc_edens_vxc_spin(rho_u, rho_d, suu, sud, sdd, edens, vrhou, vrhod, vsuu, vsud, vsdd)
    implicit none
    real(dp), intent(in)  :: rho_u, rho_d, suu, sud, sdd
    real(dp), intent(out) :: edens, vrhou, vrhod, vsuu, vsud, vsdd

    call nnxc_edens_vxc_spin_analytic(rho_u, rho_d, suu, sud, sdd, edens, vrhou, vrhod, vsuu, vsud, vsdd)
  end subroutine nnxc_edens_vxc_spin



end module xc_nn