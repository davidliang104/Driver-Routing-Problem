import com.graphhopper.jsprit.core.algorithm.VehicleRoutingAlgorithm;
import com.graphhopper.jsprit.core.algorithm.box.SchrimpfFactory;
import com.graphhopper.jsprit.core.problem.Location;
import com.graphhopper.jsprit.core.problem.VehicleRoutingProblem;
import com.graphhopper.jsprit.core.problem.job.Shipment;
import com.graphhopper.jsprit.core.problem.solution.VehicleRoutingProblemSolution;
import com.graphhopper.jsprit.core.problem.vehicle.Vehicle;
import com.graphhopper.jsprit.core.problem.vehicle.VehicleImpl;
import com.graphhopper.jsprit.core.problem.vehicle.VehicleType;
import com.graphhopper.jsprit.core.problem.vehicle.VehicleTypeImpl;
import com.graphhopper.jsprit.core.util.Solutions;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

public class Scheduler {
    public static void main(String[] args) {
        File dir = new File("output");
        // if the directory does not exist, create it
        if (!dir.exists()){
            System.out.println("creating directory ./output");
            boolean result = dir.mkdir();
            if(result) System.out.println("./output created");
        }

        /*
         * get a vehicle type-builder and build a type with the typeId "driver1" and a capacity of 2
         */
        VehicleTypeImpl.Builder vehicleTypeBuilder = VehicleTypeImpl.Builder.newInstance("car").addCapacityDimension(0,2);
        VehicleType vehicleType = vehicleTypeBuilder.build();

        /*
         * get a vehicle-builder and build a vehicle located at (40,50) with type "driver1" and a latest arrival
         * time of 1236 (which corresponds to a operation time of 1236 since the earliestStart of the vehicle is set
         * to 0 by default).
         */
        VehicleImpl.Builder vehicleBuilder = VehicleImpl.Builder.newInstance("driver1");
        vehicleBuilder.setStartLocation(Location.newInstance(19, 8));
        vehicleBuilder.setLatestArrival(1236);
        vehicleBuilder.setType(vehicleType);
        Vehicle vehicle = vehicleBuilder.build();

        /*
            Coordinates:
            Jumeirah: 19,8

            CNVL & JRSC KOC: 11, 28
            All Ratq KOC: 6, 35
         */


        /*
         * build shipments at the required locations, each with a capacity-demand of 1.
         */
        Shipment shipment1 = Shipment.Builder.newInstance("1").addSizeDimension(0,1).setPickupLocation(Location.newInstance(19,8))
                .setDeliveryLocation(Location.newInstance(11, 28)).build();
        Shipment shipment2 = Shipment.Builder.newInstance("2").addSizeDimension(0,1).setPickupLocation(Location.newInstance(19,8))
                .setDeliveryLocation(Location.newInstance(11, 28)).build();


        VehicleRoutingProblem.Builder vrpBuilder = VehicleRoutingProblem.Builder.newInstance();

        /*
         * add these shipments to the problem
         */
        vrpBuilder.addJob(shipment1).addJob(shipment2);
        VehicleRoutingProblem problem = vrpBuilder.build();

        /*
         * get the algorithm out-of-the-box.
         */
        VehicleRoutingAlgorithm algorithm = new SchrimpfFactory().createAlgorithm(problem);

        /*
         * and search a solution
         */
        Collection<VehicleRoutingProblemSolution> solutions = algorithm.searchSolutions();

        /*
         * get the best
         */
        VehicleRoutingProblemSolution bestSolution = Solutions.bestOf(solutions);

        /*
         * plot problem with solution
         */
        new GraphStreamViewer(problem).setRenderShipments(true).display();
    }
}
