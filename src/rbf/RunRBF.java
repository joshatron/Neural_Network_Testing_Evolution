package rbf;

public class RunRBF
{

    // func for testing rbf and returns avg. percent error
    public static double testRBF(double[][] buildingSet, double[][] trainingSet, double[][] testingSet, double learningRate, int numbOfClusters)
    {
        double error = 0.0;

        return error;
    }

    public static void main(String[] args)
    {
        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(3, 0.1);

        System.out.println("Hello world");
        rbfNeuralNetwork.createTrainingDataWithKClustering(500, 100000, 3);
        rbfNeuralNetwork.print();

        double x1 = Math.random() * 100;
        double x2 = Math.random() * 100;
        double x3 = Math.random() * 100;
        double[] inputs = {x1, x2, x3};
        x1 = Math.random() * 100;
        x2 = Math.random() * 100;
        x3 = Math.random() * 100;
        double[] inputs2 = {x1, x2, x3};
        x1 = Math.random() * 100;
        x2 = Math.random() * 100;
        x3 = Math.random() * 100;
        double[] inputs3 = {x1, x2, x3};
        x1 = Math.random() * 100;
        x2 = Math.random() * 100;
        x3 = Math.random() * 100;
        double[] inputs4 = {x1, x2, x3};
        x1 = Math.random() * 100;
        x2 = Math.random() * 100;
        x3 = Math.random() * 100;
        double[] inputs5 = {x1, x2, x3};

        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));

        rbfNeuralNetwork.train(10000);

        System.out.println("After Training:");

        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));
    }
}
