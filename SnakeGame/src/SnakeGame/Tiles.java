package SnakeGame;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class Tiles extends Rectangle {
    private Color color;
    private int directions;
    private int row;
    private int col;
    private int size = 20;

    public Tiles(int x, int y, int width, int height, Color c, int row, int col) {
        super(x, y, width, height);
        this.row = row;
        this.col = col;
        color = c;
    }

    public void setDirection(int direct) {
        this.directions = direct;
    }

    public int getDirection() {
        return directions;
    }

    public void setRow(int r) {
        row = r;
    }

    public void setCol(int c) {
        col = c;
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    public void move() {
        if (directions == 0) {
            x -= size;
            col--;
        }
        if (directions == 1) {
            y += size;
            row++;
        }
        if (directions == 2) {
            x += size;
            col++;
        }
        if (directions == 3) {
            y -= size;
            row--;
        }
    }

    public void draw(Graphics2D pb) {
        pb.setColor(color);
        pb.draw(this);
    }

    public void fill(Graphics2D pb) {
        pb.setColor(color);
        pb.fill(this);
    }
}
