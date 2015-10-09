package nettest;

/**
 *
 * @author david
 */
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javax.imageio.ImageIO;

public class WriteImage extends Application {

    private float[][] grid;       // Grid of values for the heatmap
    private float scalar;         // Scalar to convert grid values to colors
    private StackPane root;
    private final int iLength = 512;
    private final int jLength = 512;

    @Override
    public void start(Stage primaryStage) throws MalformedURLException, InterruptedException, ExecutionException {
        for (int i = 0; i < 3; i++) {
            int scale = 1;
            int add = 0;
            switch(i) {
                case 1:
                    scale = 2;
                    add = 1;
                    break;
                case 2:
                    scale = 5;
                    break;
                default:
                    break;
            }
            for (int j1 = 100; j1 < 500000; j1*= 10) {
                if (j1 == 100000 && i == 3) {
                    break;
                }
                int dim = 2;
                int j = j1*scale+j1*add/2;
                ImageTask(dim,j);
               
                try {
                    call();
                } catch (Exception ex) {
                    Logger.getLogger(WriteImage.class.getName()).log(Level.SEVERE, null, ex);
                }
                
                String fileName = String.format("/home/david/Programs/NeuralNets/data/images/%dD_Data_%d_vectors.png", dim, j);

                if (root != null) {
                    Scene scene = new Scene(root, iLength, jLength);
                    WritableImage write = scene.snapshot(null);
                    File outFile = new File(fileName);

                    try {
                        ImageIO.write(SwingFXUtils.fromFXImage(write, null),
                                "png", outFile);
                    } catch (IOException ex) {
                        Logger.getLogger(WriteImage.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }
        }
        System.exit(0);
    }

    public void ImageTask(int numInputs, int setSize) {
        double[][] dataset = DataTools.getDataFromFile(numInputs, setSize);
        this.grid = setGrid(dataset);
        this.scalar = getScale(dataset);
        try {
            call();
        } catch (Exception ex) {
            Logger.getLogger(WriteImage.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void call() throws Exception {

        // Create heat map
        Image heatmap = createHeatMap();

        float opacity = 1f;     // How opaque the heat map will be
        int dim = 512;          // Dimensionality of images

        // ImageView: "frame" objects for individual images
        ImageView heat = new ImageView();

        // StackPane: "canvas" where the final image is built
        root = new StackPane();

        // Scale image, set opacity, and add heat map to root pane
        if (heatmap != null) {
            heat.setImage(heatmap);
            heat.setFitWidth(dim);
            heat.setPreserveRatio(true);
            heat.setOpacity(opacity);
            root.getChildren().add(heat);
        }
    }

    // Calculate and return color for a heat map pixel
    public Color getHeatColor(int y, int x) {
        float dataPoint = grid[y][x];

/*        for (int i = 0; i < 2; i++) {// Uncomment for additional smoothing. Adjust i for more smoothing iterations.
            if (x == 0) {
                if (y == 0) {
                    dataPoint += grid[y + 1][x] + grid[y + 1][x + 1] + grid[y][x + 1];
                    dataPoint /= 4;
                } else if (y == iLength - 1) {
                    dataPoint += grid[y - 1][x] + grid[y - 1][x + 1] + grid[y][x + 1];
                    dataPoint /= 4;
                } else {
                    dataPoint += grid[y + 1][x] + grid[y + 1][x + 1] + grid[y][x + 1]
                            + grid[y - 1][x] + grid[y - 1][x + 1];
                    dataPoint /= 6;
                }
            } else if (x == jLength - 1) {
                if (y == 0) {
                    dataPoint += grid[y + 1][x] + grid[y + 1][x - 1] + grid[y][x - 1];
                    dataPoint /= 4;
                } else if (y == iLength - 1) {
                    dataPoint += grid[y - 1][x] + grid[y - 1][x - 1] + grid[y][x - 1];
                    dataPoint /= 4;
                } else {
                    dataPoint += grid[y + 1][x] + grid[y + 1][x - 1] + grid[y][x - 1]
                            + grid[y - 1][x] + grid[y - 1][x - 1];
                    dataPoint /= 6;
                }
            } else if (y == 0) {
                dataPoint += grid[y + 1][x] + grid[y + 1][x - 1] + grid[y][x - 1]
                        + grid[y + 1][x + 1] + grid[y][x + 1];
                dataPoint /= 6;
            } else if (y == iLength - 1) {
                dataPoint += grid[y - 1][x] + grid[y - 1][x - 1] + grid[y][x - 1]
                        + grid[y - 1][x + 1] + grid[y][x + 1];
                dataPoint /= 6;
            } else {
                dataPoint += grid[y - 1][x] + grid[y - 1][x - 1] + grid[y][x - 1]
                        + grid[y - 1][x + 1] + grid[y][x + 1] + grid[y + 1][x]
                        + grid[y + 1][x + 1] + grid[y + 1][x - 1];
                dataPoint /= 9;
            }
        }
*/
        dataPoint *= scalar;
        float saturation = 1f;
        float brightness = 1f;
        if (dataPoint > 255f) {
            saturation -= 1f - 255f / dataPoint;
            brightness -= 1f - 255f / dataPoint;
            dataPoint = 255f;
        } else if (dataPoint < 4f && dataPoint > 0) {
            dataPoint = 4f - dataPoint;
            dataPoint *= -4f;
        }
        dataPoint = 255f - dataPoint;
        Color color = Color.hsb(dataPoint, saturation, brightness);
        return color;
    }

    // Create heat map from grid[][] values
    public WritableImage createHeatMap() {
        // Create WritableImage
        WritableImage makeImage = new WritableImage(iLength, jLength);

        PixelWriter pixelWriter = makeImage.getPixelWriter();

        // Write the color of each pixel
        for (int yLoc = 0; yLoc < makeImage.getHeight(); yLoc++) {
            for (int xLoc = 0; xLoc < makeImage.getWidth(); xLoc++) {
                Color color = getHeatColor(yLoc, xLoc);

                // Set the color of 4x4 pixel grid where top left = (readX, readY
                pixelWriter.setColor(xLoc, yLoc, color);
            }
        }
        return makeImage;
    }

    // Retrieve StackPane object from this thread
    public StackPane getPane() {
        return root;
    }

    public float getScale(double dataset[][]) {
        double max = 0;
        for (int i = 0; i < dataset.length; i++) {
            double next = dataset[i][dataset[i].length - 1];
            if (next > max) {
                max = next;
            }
        }

        float scale = 256.0f / ((float) max / 4f );

        return scale;
    }

    public float[][] setGrid(double[][] dataset) {


        float domainSize = DataTools.getDomainSize();
        float domainSizeHalf = domainSize / 2;

        float[][] newGrid = new float[iLength][jLength];

        for (int i = 0; i < dataset.length; i++) {
            int x = (int) ((dataset[i][0] + domainSizeHalf) * (float) jLength / domainSize);
            if (x >= jLength) {
                x = jLength - 1;
            }
            int y = (int) ((dataset[i][1] + domainSizeHalf) * (float) iLength / domainSize);
            if (y >= iLength) {
                y = iLength - 1;
            }

            float value = newGrid[y][x];
            if (value == 0 || value > dataset[i][dataset[i].length-1]) {    // if empty or value is bigger than another possible value
                value = (float) dataset[i][dataset[i].length-1];
            } 
            
            newGrid[y][x] = value;
        }

        for (int i = 0; i < dataset.length; i++) {
            int x = (int) ((dataset[i][0] + domainSizeHalf) * (float) jLength / domainSize);
            if (x >= jLength) {
                x = jLength - 1;
            }
            int y = (int) ((dataset[i][1] + domainSizeHalf) * (float) iLength / domainSize);
            if (y >= iLength) {
                y = iLength - 1;
            }

            if (y > 0 && y < iLength - 1 && x > 0 && x < jLength - 1) {
                if (newGrid[y + 1][x + 1] == 0) {
                    newGrid[y + 1][x + 1] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y + 1][x] == 0) {
                    newGrid[y + 1][x] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y + 1][x - 1] == 0) {
                    newGrid[y + 1][x - 1] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y][x + 1] == 0) {
                    newGrid[y][x + 1] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y][x - 1] == 0) {
                    newGrid[y][x - 1] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y - 1][x + 1] == 0) {
                    newGrid[y - 1][x + 1] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y - 1][x] == 0) {
                    newGrid[y - 1][x] = (float) dataset[i][dataset[i].length-1];
                }
                if (newGrid[y - 1][x - 1] == 0) {
                    newGrid[y - 1][x - 1] = (float) dataset[i][dataset[i].length-1];
                }
            }
        }

        if (newGrid[0][0] == 0) {
            newGrid[0][0] = newGrid[0][1] + newGrid[1][1] + newGrid[1][0];
            newGrid[0][0] /= 3;
        }

        if (newGrid[iLength - 1][0] == 0) {
            newGrid[iLength - 1][0] = newGrid[iLength - 1][1] + newGrid[iLength - 2][1] + newGrid[iLength - 2][0];
            newGrid[iLength - 1][0] /= 3;
        }

        if (newGrid[0][jLength - 1] == 0) {
            newGrid[0][jLength - 1] = newGrid[1][jLength - 1] + newGrid[1][jLength - 2] + newGrid[0][jLength - 2];
            newGrid[0][jLength - 1] /= 3;
        }

        if (newGrid[iLength - 1][jLength - 1] == 0) {
            newGrid[iLength - 1][jLength - 1] = newGrid[iLength - 1][jLength - 2] + newGrid[iLength - 2][jLength - 1] + newGrid[iLength - 2][jLength - 2];
            newGrid[iLength - 1][jLength - 1] /= 3;
        }

        for (int i = 1; i < iLength - 1; i++) {

            if (newGrid[0][i] == 0) {
                newGrid[0][i] = newGrid[0][i - 1] + newGrid[0][i + 1]
                        + newGrid[1][i - 1] + newGrid[1][i] + newGrid[1][i + 1];
                newGrid[0][i] /= 5;
            }

            if (newGrid[iLength - 1][i] == 0) {
                newGrid[iLength - 1][i] = newGrid[iLength - 2][i - 1] + newGrid[iLength - 2][i] + newGrid[iLength - 2][i + 1]
                        + newGrid[iLength - 1][i - 1] + newGrid[iLength - 1][i + 1];
                newGrid[iLength - 1][i] /= 5;
            }

            if (newGrid[i][0] == 0) {
                newGrid[i][0] = newGrid[i - 1][0] + newGrid[i + 1][0]
                        + newGrid[i - 1][1] + newGrid[i][1] + newGrid[i + 1][1];
                newGrid[i][0] /= 5;
            }

            if (newGrid[i][jLength - 1] == 0) {
                newGrid[i][jLength - 1] = newGrid[i - 1][jLength - 2] + newGrid[i][jLength - 2] + newGrid[i + 1][jLength - 2]
                        + newGrid[i - 1][jLength - 1] + newGrid[i + 1][jLength - 1];
                newGrid[i][jLength - 1] /= 5;
            }

            for (int j = 1; j < jLength - 1; j++) {
                if (newGrid[i][j] == 0) {
                    newGrid[i][j] = newGrid[i - 1][j - 1] + newGrid[i - 1][j] + newGrid[i - 1][j + 1]
                            + newGrid[i][j - 1] + newGrid[i][j + 1]
                            + newGrid[i + 1][j - 1] + newGrid[i + 1][j] + newGrid[i + 1][j + 1];
                    newGrid[i][j] /= 8;
                }
            }
        }

        return newGrid;
    }

    public static void main(String[] args) throws MalformedURLException {
        launch(args);
    }
}
