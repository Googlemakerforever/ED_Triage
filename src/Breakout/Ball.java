package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class Ball extends Rectangle {

    private int x, y;
    private int diameter;

    private int dx = 4;
    private int dy = 4;

    
    private final int defaultDx = 4;
    private final int defaultDy = 4;

    public Ball(int x, int y, int diameter) {
        this.x = x;
        this.y = y;
        this.diameter = diameter;
    }

    public void move() {
        x += dx;
        y += dy;
    }

    public void bounceX() {
        dx = -dx;
    }

    public void bounceY() {
        dy = -dy;
    }

    public void slowDown() {
        dx = reduceSpeed(dx);
        dy = reduceSpeed(dy);
    }

    private int reduceSpeed(int value) {
        int direction = value < 0 ? -1 : 1;
        int magnitude = Math.abs(value);
        if (magnitude > 2) {
            magnitude -= 1;
        }
        return magnitude * direction;
    }

    
    public void resetSpeed() {
        dx = defaultDx;
        dy = defaultDy;

        
        if (dy > 0) dy = -dy;
    }

    public int getXPos() { return x; }
    public int getYPos() { return y; }
    public int getDiameter() { return diameter; }

    public int getDx() { return dx; }
    public int getDy() { return dy; }

    
    public void setXPos(int x) { this.x = x; }
    public void setYPos(int y) { this.y = y; }

    public void draw(Graphics2D win) {
        win.setColor(Color.GREEN);
        win.fillOval(x, y, diameter, diameter);
    }

    @Override
    public Rectangle getBounds() {
        return new Rectangle(x, y, diameter, diameter);
    }
}
