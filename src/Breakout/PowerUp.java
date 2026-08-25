package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class PowerUp {

    public enum Type {
        RED,
        YELLOW,
        GREEN
    }

    private int x, y, width, height;
    private int dy = 4;
    private boolean active = true;
    private Type type;

    public PowerUp(int x, int y, int width, int height, Type type) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.type = type;
    }

    public void move() {
        if (active) {
            y += dy;
        }
    }

    public void draw(Graphics2D win) {
        if (!active) return;

        if (type == Type.RED) {
            win.setColor(Color.RED);
        } else if (type == Type.GREEN) {
            win.setColor(Color.GREEN);
        } else {
            win.setColor(Color.YELLOW);
        }
        win.fillRect(x, y, width, height);
        win.setColor(Color.BLACK);
        win.drawRect(x, y, width, height);
    }

    public Rectangle getBounds() {
        return new Rectangle(x, y, width, height);
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public int getY() {
        return y;
    }

    public Type getType() {
        return type;
    }
}
