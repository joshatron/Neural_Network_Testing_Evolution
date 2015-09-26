package rbf;

public class RunRBF
{
    public static void main(String[] args)
    {
        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork();

        rbfNeuralNetwork.createTrainingData(1000);


        double x1 = Math.random() * 100;
        double x2 = Math.random() * 100;
        double x3 = Math.random() * 100;
        double[] inputs = {x1, x2, x3};

        System.out.println("Results: ");
        System.out.println(rbfNeuralNetwork.findCorrectAnswer(inputs));
    }
}
