package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.Random;

public class BrickParticle {

    private int x, y, w, h;
    private int dx, dy;
    private Color color;

    private static final Random rng = new Random();

    public BrickParticle(int x, int y, int w, int h, Color color) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.color = color;

        
        this.dx = 1;
        this.dy = -1;
    }

    public void move() {
        
        if (Math.random() < 0.5) dx = 1;
        else dx = -1;

        if (Math.random() < 0.5) dy = 1;
        else dy = -1;

        x += dx;
        y += dy;
    }

    public void draw(Graphics2D win) {
        win.setColor(color);
        win.fillRect(x, y, w, h);
    }
}
