from netgen.occ import MoveTo, Box, Glue, OCCGeometry, X, Y, Z
# It seems that the import of netgen.gui is necessary for the GUI to start
import netgen.gui # pylint: disable=unused-import
import ngsolve
import matplotlib.pyplot as plt
import numpy as np
from rf_solver import SolverRF

class IfaAntennaCase:
    def __init__(self):
        self.geom = self.create_geometry()
        self.mesh = self.generate_mesh()
        self.frequencies = np.arange(2.1e9, 2.85e9, 0.05e9)
        # reference s11 calculated for np.arange(2.1e9, 2.85e9, 0.05e9) frequencies with cenos
        # calculation was done for approx 600 k DOF's (73 k elements with second order finite element space)
        self.reference_s11_2nd_order = [
            -2.986093,
            -3.962175,
            -5.265836,
            -6.990170,
            -9.236308,
            -12.04847,
            -15.00001,
            -16.16484,
            -14.61554,
            -12.46401,
            -10.67441,
            -9.288600,
            -8.204403,
            -7.330339,
            -6.599893,
        ]
        self.reference_s11 = [
            -2.475688025,
            -3.222148218,
            -4.202493369,
            -5.481446021,
            -7.135602047,
            -9.250379552,
            -11.88238368,
            -14.80542545,
            -16.62158671,
            -15.69395906,
            -13.55988979,
            -11.61008927,
            -10.05254568,
            -8.814651333,
            -7.807999303,
        ]
        self.s_params = []
        self.domains = {"dielectric": {"type": "dielectric", "epsilon": 2.2}}
        self.boundaries = {"air_outer": {"type": "outer"}, "ground": {"type": "pec"}, "patch": {"type": "pec"}}
        self.lumped_elements = {"loading_pin": {"type": "feed"}}

    def create_geometry(self):
        cm = 1e-2
        mm = 1e-3
        um = 1e-6

        box_y = 32 * mm
        box_x = 45 * mm

        ground_x = 35 * mm

        start_y = 2.5 * mm
        wx = 6 * mm
        wy = 26 * mm

        gw = 7 * mm
        feed_gap = 0.8 * mm
        line_width = 1 * mm
        substrate_height = 1.6 * mm

        air_delta = 36 * mm
        copper_thickness = 35 * um

        antenna_part = (
            MoveTo(ground_x, start_y)
            .Line(wx+line_width)
            .Rotate(90)
            .Line(wy)
            .Rotate(90)
            .Line(line_width)
            .Rotate(90)
            .Line(wy - gw - line_width)
            .Rotate(-90)
            .Line(wx - feed_gap)
            .Rotate(90)
            .Line(line_width)
            .Rotate(90)
            .Line(wx - feed_gap)
            .Rotate(-90)
            .Line(gw - line_width)
            .Rotate(-90)
            .Line(wx)
            .Close()
            .Face()
            .Move((0, 0, substrate_height))
        )
        antenna_part = antenna_part.Extrude(copper_thickness)

        ground_copper = MoveTo(0,0).Rectangle(ground_x, box_y).Face().Move((0, 0, substrate_height))
        ground_copper = ground_copper.Extrude(copper_thickness)

        dielectric = MoveTo(0,0).Rectangle(box_x, box_y).Face().Extrude(substrate_height)

        pins = MoveTo(ground_x, start_y + gw + line_width/2).Rectangle(feed_gap, line_width/2).Face().Move((0, 0, substrate_height))
        pins.edges.Min(Y).name = "loading_pin"

        air_box = Box(
            (-air_delta, -air_delta, -air_delta),
            (air_delta + box_x, air_delta + box_y, air_delta + substrate_height),
        )
        air_box.faces.name = "air_outer"
        air_box = air_box - dielectric
        air_box.faces.col = (0, 0, 1, 0.3)
        air_box.solids.name = "air"
        ground_copper.faces.name = "ground"
        antenna_part.faces.name = "patch"
        dielectric.solids.name = "dielectric"
        dielectric.faces.col = (1, 1, 0)

        shape = Glue([antenna_part, ground_copper, dielectric, pins])

        shape = Glue([shape, air_box])

        geom = OCCGeometry(shape)
        ngsolve.Draw(geom)
        return geom

    def generate_mesh(self):
        ngmesh = self.geom.GenerateMesh(maxh=0.01, segmentsperedge=0)
        mesh = ngsolve.Mesh(ngmesh)
        ngsolve.Draw(mesh)
        return mesh

    def calculate(self, refinement_function = None):
        # reset refinement
        for el in self.mesh.Elements(ngsolve.VOL):
            self.mesh.SetRefinementFlag(el, False)

        self.s_params = []
        for freq in self.frequencies:
            solver = SolverRF(self.mesh, self.domains, self.boundaries, self.lumped_elements, {"frequency": freq})
            solver.assemble()
            solver.solve()
            solver.finalize()
            self.s_params.append(solver.s11)
            # we set refinement flags for next call to refine after each frequency
            if refinement_function is not None:
                refinement_function(self.mesh, solver)

    def refine_mesh(self):
        self.mesh.Refine()

if __name__ == "__main__":
    # create instance of ifa antenna case
    ifa_antenna_case = IfaAntennaCase()
    ngsolve.Draw(ifa_antenna_case.mesh)

    def refinement_func(mesh, solver):
        """Simple function defined to refine elements with high electric field"""
        solver_fes = solver.electric_field.space
        max_dof = np.max(solver.electric_field.vec)
        for el in mesh.Elements(ngsolve.VOL):
            for edge in el.edges:
                for dof in solver_fes.GetDofNrs(edge):
                    if solver.electric_field.vec[dof] > 0.8 * max_dof:
                        mesh.SetRefinementFlag(el, True)
    
    def refinement_function2(mesh, solver):
        errors = solver.estimate_error()
        max_error = np.max(errors)
        for idx, el in enumerate(mesh.Elements(ngsolve.VOL)):
            if errors[idx] > 0.8 * max_error:
                mesh.SetRefinementFlag(el, True)


    # now let's use the function in the calculate method
    num_iterations = 6
    s_params_list = []
    norm_residuals_list = []

    # here we loop over iterations and refine mesh each time.
    # we also pass our refinement function to calculate method to set refinement flags for mesh elements
    for i in range(num_iterations):
        ifa_antenna_case.calculate(refinement_function2)
        s_params = ifa_antenna_case.s_params
        s_params_list.append(s_params)
        
        residuals = np.array(s_params) - np.array(ifa_antenna_case.reference_s11)
        norm_residuals = np.linalg.norm(residuals)
        norm_residuals_list.append(norm_residuals)
        print(f"Residuals after calculation {i + 1}: ", norm_residuals)
        ifa_antenna_case.refine_mesh()

    # Plotting the results
    plt.figure()
    for i, s_params in enumerate(s_params_list):
        plt.plot(ifa_antenna_case.frequencies, s_params, label=f"Iteration {i}")
    plt.plot(ifa_antenna_case.frequencies, ifa_antenna_case.reference_s11, linewidth=2.5, label="Reference")
    plt.legend()
    plt.show()
    ngsolve.Draw(ifa_antenna_case.mesh)

