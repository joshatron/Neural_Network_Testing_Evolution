package rbf;

public class RunRBF
{
    public static void main(String[] args)
    {
//        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(3, 0.05);

//        System.out.println("Hello world");
//        rbfNeuralNetwork.createTrainingDataWithKClustering(500, 100000, 3);
//        rbfNeuralNetwork.print();
//
//        double x1 = Math.random() * 100;
//        double x2 = Math.random() * 100;
//        double x3 = Math.random() * 100;
//        double[] inputs = {x1, x2, x3};
//        x1 = Math.random() * 100;
//        x2 = Math.random() * 100;
//        x3 = Math.random() * 100;
//        double[] inputs2 = {x1, x2, x3};
//        x1 = Math.random() * 100;
//        x2 = Math.random() * 100;
//        x3 = Math.random() * 100;
//        double[] inputs3 = {x1, x2, x3};
//        x1 = Math.random() * 100;
//        x2 = Math.random() * 100;
//        x3 = Math.random() * 100;
//        double[] inputs4 = {x1, x2, x3};
//        x1 = Math.random() * 100;
//        x2 = Math.random() * 100;
//        x3 = Math.random() * 100;
//        double[] inputs5 = {x1, x2, x3};
//
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));
//
//        rbfNeuralNetwork.train(10000);
//
//        System.out.println("After Training:");
//
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
//        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
//        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));


        // training size
//        for (int i = 1000; i <= 1000000; i *= 10)
//        {
//            // cluster size
//            for (int j = 10; j < 1000; j+=50)
//            {
//                RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(3, 0.05);
//                rbfNeuralNetwork.createTrainingDataWithKClustering(j, 100000, 3);
//                rbfNeuralNetwork.train(i);
//                double percentage = percentageOnNumb(rbfNeuralNetwork, 100, 3);
//
//                System.out.println("Percent Error: " + percentage + ", for training size:" + i + ", clusters: " + j);
//            }
//        }

        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(3, 0.05);
        rbfNeuralNetwork.createTrainingDataWithKClustering(500, 100000, 3);
        rbfNeuralNetwork.train(5000000);
        double percentage = percentageOnNumb(rbfNeuralNetwork, 100, 3);

        System.out.println("Percent Error: " + percentage + ", for training size:" + 5000000 + ", clusters: " + 500);

    }

    public static double getPercentage(double est, double act)
    {
        return ((est - act) / act) * 100;
    }

    public static double percentageOnNumb(RBFNeuralNetwork rbfNeuralNetwork, int numb, int n)
    {
        double totalError = 0.0;
        for (int i = 0; i < numb; i++)
        {
            double[] inputs = new double[n];
            for (int j = 0; j < n; j++)
            {
               inputs[j] = Math.random() * 100;
            }

            double est = rbfNeuralNetwork.getResult(inputs);
            double act = rbfNeuralNetwork.findCorrectAnswer(inputs);
            totalError += getPercentage(est, act);
        }
        totalError /= numb;
        return totalError;
    }


}
