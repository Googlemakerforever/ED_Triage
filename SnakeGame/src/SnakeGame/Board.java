package SnakeGame;

import java.awt.Color;
import java.awt.Graphics2D;

public class Board {
    private Tiles[][] grid = new Tiles[40][60];

    public Board() {
        int x = 0;
        int y = 0;
        int size = 20;
        for (int i = 0; i < 40; i++) {
            for (int j = 0; j < 60; j++) {
                x += size;
                grid[i][j] = new Tiles(x, y, size, size, Color.GRAY, i, j);
            }
            x = 0;
            y += size;
        }
    }

    public Tiles[][] getGrid() {
        return grid;
    }

    public void draw(Graphics2D pb) {
        for (Tiles[] i : grid) {
            for (Tiles t : i) {
                t.draw(pb);
            }
        }
    }
}
