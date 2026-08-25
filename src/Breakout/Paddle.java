package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class Paddle {

    private int x, y, width, height;
    private int speed = 10;

    public Paddle(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public void moveLeft() {
        x -= speed;
        if (x < 0) x = 0;
    }

    public void moveRight(int windowWidth) {
        x += speed;
        if (x + width > windowWidth) {
            x = windowWidth - width;
        }
    }

    public void increaseSpeed(int amount) {
        speed += amount;
        if (speed > 25) { 
            speed = 25;
        }
    }

    public void increaseWidth(int amount, int windowWidth) {
        width += amount;
        int maxWidth = 220;
        if (width > maxWidth) {
            width = maxWidth;
        }
        if (x + width > windowWidth) {
            x = windowWidth - width;
        }
        if (x < 0) {
            x = 0;
        }
    }

    public void draw(Graphics2D win) {
        win.setColor(Color.BLUE);
        win.fillRect(x, y, width, height);
    }

    public Rectangle getBounds() {
        return new Rectangle(x, y, width, height);
    }

    public int getY() {
        return y;
    }
}
