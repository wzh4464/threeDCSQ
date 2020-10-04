import functional_func.draw_func as draw_pack
import functional_func.spherical_func as sphe_pack


def main():
    print("start cell shape analysis")
    spherical_fibonacci = sphe_pack.fibonacci_sphere(500, 10)
    draw_pack.draw_3D_curve(spherical_fibonacci)
    draw_pack.draw_3D_points(spherical_fibonacci)



if __name__ == '__main__':
    main()
    # print(__name__)
