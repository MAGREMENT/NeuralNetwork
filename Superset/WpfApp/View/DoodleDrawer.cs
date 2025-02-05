using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace WpfApp.View;

public class DoodleDrawer : FrameworkElement
{
    public int Rows { get; set; } = 28;
    public int Columns { get; set; } = 28;

    private double[,] _data = new double[28, 28];

    public event Action<int, int, double, double>? OnDraw;

    public void SetData(double[,] data)
    {
        _data = data;
        Refresh();
    }
    
    // Provide a required override for the VisualChildrenCount property.
    protected override int VisualChildrenCount => 0;

    // Provide a required override for the GetVisualChild method.
    protected override Visual GetVisualChild(int index)
    {
        return null!;
    }
    
    public void Refresh()
    {
        Dispatcher.Invoke(InvalidateVisual);
    }
    
    protected override void OnRender(DrawingContext context)
    {
        if (double.IsNaN(ActualWidth) || ActualWidth == 0 ||
            double.IsNaN(ActualHeight) || ActualWidth == 0) return;
        
        var w = ActualWidth / Columns;
        var h = ActualHeight / Rows;

        for (int c = 0; c < Columns; c++)
        {
            for (int r = 0; r < Rows; r++)
            {
                var val = (byte)(255 * (1 - _data[c, r]));
                var col = new Color
                {
                    A = 255,
                    B = val,
                    G = val,
                    R = val
                };
                
                context.DrawRectangle(new SolidColorBrush(col), null,
                    new Rect(c * w, r * h, w, h));
            }
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (e.LeftButton == MouseButtonState.Pressed) Draw(e);
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        Draw(e);
    }

    private void Draw(MouseEventArgs e)
    {
        var pos = e.GetPosition(this);
        OnDraw?.Invoke((int)(pos.X / ActualWidth * Columns), (int)(pos.Y / ActualHeight * Rows), 0.5, 0.5);
    }
}