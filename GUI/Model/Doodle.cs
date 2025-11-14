namespace Model;

using Base;

public class Doodle(int _width, int height)
{
    private readonly double[] _current = new double[_width * height];

    public double this[int col, int row] => _current[col * _width + row];

    public void Draw(int col, int row, double colPercent, double rowPercent)
    {
        _current[row * _width + col] = 1;
    }

    public void SetData(double[] d)
    {
        d.CopyTo(_current, 0);
    }

    public void Clear()
    {
        Array.Fill(_current, 0);
    }

    public double[] ToNeuralNetworkInputs() => _current.Copy();

    public double[,] To2DData() => _current.To2D(_width, _current.Length / _width);
}