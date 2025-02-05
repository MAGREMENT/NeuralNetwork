using Model;

namespace WpfApp.Presenter;

public class MainPresenter(IMainView _view)
{
    private readonly Doodle _doodle = new(28, 28);
    private readonly IReadOnlyList<(int, double[])> dataSet = MNIST.Read(
        @"C:\Users\Zach\Desktop\Perso\NeuralNetwork\Superset\Model\mnist-data\t10k-labels.idx1-ubyte", 
        @"C:\Users\Zach\Desktop\Perso\NeuralNetwork\Superset\Model\mnist-data\t10k-images.idx3-ubyte", 100);

    private int _index = -1;

    public void Next()
    {
        if (_index >= dataSet.Count - 1) return;
        _index++;
        _view.SetDoodleData(dataSet[_index].Item2.To2D(28, 28));
    }

    public void Previous()
    {
        if (_index <= 0) return;
        _index--;
        _view.SetDoodleData(dataSet[_index].Item2.To2D(28, 28));
    }

    public void Draw(int col, int row, double colPercent, double rowPercent)
    {
        _doodle.Draw(col, row, colPercent, rowPercent);
        _view.SetDoodleData(_doodle.To2DData());
    }
}

public interface IMainView
{
    void SetDoodleData(double[,] data);
}