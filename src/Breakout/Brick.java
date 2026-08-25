package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class Brick {

    private int x, y, width, height;
    private boolean visible = true;

    
    private BrickParticle[] particles;
    private boolean partVis = false;
    private boolean counting = false;
    private int count = 0;

    
    private static final int COLS = 26;
    private static final int ROWS = 6;

    public Brick(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;

        buildParticles();
    }

    private void buildParticles() {
        particles = new BrickParticle[COLS * ROWS];

        int pW = Math.max(1, width / COLS);
        int pH = Math.max(1, height / ROWS);

        int idx = 0;
        int curX = x;
        int curY = y;

        for (int i = 0; i < COLS * ROWS; i++) {
            particles[idx++] = new BrickParticle(curX, curY, pW, pH, Color.WHITE);

            curX += pW;

            if ((i + 1) % COLS == 0) {
                curX = x;
                curY += pH;
            }
        }
    }

    
    public void hit() {
        if (!visible) return;

        partVis = true;
        counting = true;
        count = 0;
    }

    
    public void update() {
        if (partVis) {
            
            for (BrickParticle p : particles) {
                p.move();
            }
        }

        if (counting) {
            count++;

            
            if (count >= 60) {
                partVis = false;
                counting = false;
                visible = false; 
            }
        }
    }

    public void draw(Graphics2D win) {
        
        if (partVis) {
            for (BrickParticle p : particles) {
                p.draw(win);
            }
            return;
        }

        
        if (visible) {
            win.setColor(Color.RED);
            win.fillRect(x, y, width, height);
            win.setColor(Color.BLACK);
            win.drawRect(x, y, width, height);
        }
    }

    public Rectangle getBounds() {
        return new Rectangle(x, y, width, height);
    }

    public boolean isVisible() {
        return visible;
    }

    public int getX() { return x; }
    public int getY() { return y; }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
}
