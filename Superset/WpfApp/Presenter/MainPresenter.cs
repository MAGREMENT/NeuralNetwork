using Model;

namespace WpfApp.Presenter;

public class MainPresenter
{
    private readonly IMainView _view;
    
    private readonly Doodle _doodle = new(28, 28);
    private readonly IReadOnlyList<(int, double[])> dataSet = MNIST.Read(
        "mnist-data/t10k-labels.idx1-ubyte", 
        "mnist-data/t10k-images.idx3-ubyte", 100);
    private readonly NeuralNetwork _network = new(784, 200, 100, 9);
    private int _index = -1;

    public MainPresenter(IMainView view)
    {
        _view = view;
        _network.Randomize(0, 1);
    }

    public void Next()
    {
        if (_index >= dataSet.Count - 1) return;
        _index++;
        
        _doodle.SetData(dataSet[_index].Item2);
        _view.SetDoodleData(dataSet[_index].Item2.To2D(28, 28));
        Predict();
    }

    public void Previous()
    {
        if (_index <= 0) return;
        _index--;

        _doodle.SetData(dataSet[_index].Item2);
        _view.SetDoodleData(dataSet[_index].Item2.To2D(28, 28));
        Predict();
    }

    public void Draw(int col, int row, double colPercent, double rowPercent)
    {
        _doodle.Draw(col, row, colPercent, rowPercent);
        _view.SetDoodleData(_doodle.To2DData());
    }

    public void Predict()
    {
        var output = _network.Predict(_doodle.ToNeuralNetworkInputs());

        var predictions = new List<(int, double)>(9);
        for (int i = 0; i < output.Length; i++)
        {
            predictions.Add((i + 1, Math.Round(output[i] * 100, 2)));
        }
        
        predictions.Sort((a, b) => a.Item2.CompareTo(b.Item2));
        _view.SetPredictions(predictions);
    }
}

public interface IMainView
{
    void SetDoodleData(double[,] data);
    void SetPredictions(IReadOnlyList<(int, double)> predictions);
}