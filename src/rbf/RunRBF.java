package rbf;

public class RunRBF
{
    // func for testing rbf and returns avg. percent error
    // rbfNeuralNetwork.getResults(double[] inputs); returns networks value
    // rbfNeuralNetwork.findCorrectAnswer(double[] inputs); returns actual value
    // sizes of training data: 1,000 50,000 100,000 500,000 1,000,000 5,000,000
    // sizes of building set: 500 10,000 and 100,000
    // learning rates 0.1,0.05,0.01,0.005,0.001
    public static RBFNeuralNetwork testRBF(double[][] buildingSet, double[][] trainingSet, double learningRate, int numbOfClusters, int basisFunction, int repeats)
    {
        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(buildingSet[0].length - 1, learningRate, basisFunction, repeats);
        rbfNeuralNetwork.createTrainingDataWithKClustering(numbOfClusters, buildingSet);
        rbfNeuralNetwork.train(trainingSet);
        return rbfNeuralNetwork;
    }

    public static void main(String[] args)
    {
        double[] errors;

        RBFNeuralNetwork rbfNeuralNetwork2 = new RBFNeuralNetwork(5, 0.01, 0, 100);
//        double[][] buildingData = new double[250000][3];
//        double[][] traingSet = new double[250000][3];
//
//        for (int i = 0; i < buildingData.length; i++)
//        {
//            for (int j = 0; j < buildingData[0].length; j++)
//            {
//                if (j == (buildingData[0].length - 1))
//                {
//                    double[] temp = {buildingData[i][0], buildingData[i][1]};
//                    double[] temp2 = {traingSet[i][0], traingSet[i][1]};
//                    buildingData[i][j] = rbfNeuralNetwork2.findCorrectAnswer(temp);
//                    traingSet[i][j] = rbfNeuralNetwork2.findCorrectAnswer(temp2);
//                }
//                else
//                {
//                    buildingData[i][j] = Math.random() * 10 - 5;
//                    traingSet[i][j] = Math.random() * 10 - 5;
//                }
//            }
//        }
//
//        rbfNeuralNetwork2.createTrainingDataWithKClustering(500, buildingData);

        rbfNeuralNetwork2.createRandomTrainingDataWithKClustering(500, 100000);

        errors = percentageOnNumb(rbfNeuralNetwork2, 50, 5);
        System.out.println("");
        System.out.println("Using predetermined building and training data");
        System.out.println("Error percentage: " + errors[0]);
        System.out.println("Total Error: " + errors[1]);
        System.out.println("Squared Error: " + errors[2]);

        rbfNeuralNetwork2.trainRandom(100);

        errors = percentageOnNumb(rbfNeuralNetwork2, 10000, 5);
        System.out.println("");
        System.out.println("Using predetermined building and training data");
        System.out.println("Error percentage: " + errors[0]);
        System.out.println("Total Error: " + errors[1]);
        System.out.println("Squared Error: " + errors[2]);
    }


    /**
     * This method returns the percentage error for two data points
     */
    public static double getPercentage(double est, double act)
    {
        return ((est - act) / act) * 100;
    }

    /**
     * Returns array [% error, error, squared error]
     */
    public static double[] percentageOnNumb(RBFNeuralNetwork rbfNeuralNetwork, int numb, int n)
    {
        double[] errors;
        double totalError = 0.0;
        double error = 0.0;
        double squaredError = 0.0;
        for (int i = 0; i < numb; i++)
        {
            double[] inputs = new double[n];
            for (int j = 0; j < n; j++)
            {
               inputs[j] = Math.random() * 10 - 5;
            }

            double est = rbfNeuralNetwork.getResult(inputs);
            double act = rbfNeuralNetwork.findCorrectAnswer(inputs);
            totalError += Math.abs(getPercentage(est, act));
            error += Math.abs(est - act);
            squaredError += (est - act) * (est - act);
        }
        totalError /= numb;

        errors = new double[] {totalError, error, squaredError};
        return errors;
    }
}
