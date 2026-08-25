package game;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

/**
 * @(#)GameDriverV4.java
 *
 * updates V4: JFrame included, keylistener included, switch to render, game loop
 * from tasktimer to thread
 * Updates V5: keyTyped, switched to update and render
 * Updates V6: Added Pong game implementation
 *
 * @version 6.0 9/24/2025
 */
public abstract class GDV5 extends Canvas implements Runnable, KeyListener {
    private static final long serialVersionUID = 1L;

    private int framesPerSecond;
    public static boolean[] KeysPressed;
    // default window sizes
    private static int MAX_WINDOW_X = 650;
    private static int MAX_WINDOW_Y = 750;
    private static int PADDING = 1;
    // it is your responsibility to handle the release on keysTyped
    public static boolean[] KeysTyped;

    private JFrame frame;
    private String title = "Pong";
    private boolean cleanCanvas = true;
    private volatile boolean running = false;
    private Thread gameThread;

    /**
     *
     * @param frames
     */
    public GDV5(int frames) {
        this.framesPerSecond = frames;
        this.addKeyListener(this);
        // allocate arrays slightly larger than KEY_LAST to avoid index issues
        KeysPressed = new boolean[KeyEvent.KEY_LAST + 1];
        KeysTyped = new boolean[KeyEvent.KEY_LAST + 1];
        this.setBackground(Color.BLACK);
        this.setPreferredSize(new Dimension(MAX_WINDOW_X, MAX_WINDOW_Y));
    }

    public GDV5() {
        this(100); // default setting
    }

    public void start() {
        if (running) {
            return;
        }

        // ensure canvas has a size
        if (this.getWidth() == 0 || this.getHeight() == 0) {
            this.setPreferredSize(new Dimension(MAX_WINDOW_X, MAX_WINDOW_Y));
        }

        frame = new JFrame();
        frame.add(this);
        frame.pack();
        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setResizable(false);
        frame.setVisible(true);
        this.startThread();
    }

    private synchronized void startThread() {
        if (running) {
            return;
        }
        running = true;
        gameThread = new Thread(this, "gdv5-game-loop");
        gameThread.start(); // calls run method after paint
        this.setFocusable(true);
        this.requestFocus();
    }

    public void setFrames(int num) {
        this.framesPerSecond = num;
    }

    public abstract void update();

    public abstract void draw(Graphics2D win);

    private void render() {
        BufferStrategy buffs = this.getBufferStrategy();
        if (buffs == null) {
            this.createBufferStrategy(3);
            buffs = this.getBufferStrategy();
        }
        Graphics g = buffs.getDrawGraphics();
        if (this.cleanCanvas) {
            g.setColor(this.getBackground());
            g.fillRect(0, 0, this.getWidth(), this.getHeight());
        }
        draw((Graphics2D) g);
        g.dispose();
        buffs.show();
    }

    @Override
    public void run() {
        long lastTime = System.nanoTime();
        // correct conversion: nanoseconds per frame
        double nanoSecondConversion = 1_000_000_000.0 / (double) this.framesPerSecond;
        double changeInSeconds = 0;

        while (running) {
            long now = System.nanoTime();
            changeInSeconds += (now - lastTime) / nanoSecondConversion;

            boolean updated = false;
            while (changeInSeconds >= 1) {
                update();
                changeInSeconds--;
                updated = true;
            }

            if (updated) {
                render();
            } else {
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    running = false;
                }
            }

            lastTime = now;
        }
    }

    public BufferedImage addImage(String name) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(this.getClass().getResourceAsStream(name));
        } catch (IOException | IllegalArgumentException e) {
            System.out.println("Error loading image: " + name + " -> " + e.getMessage());
        }
        return img;
    }

    @Override
    public void keyPressed(KeyEvent e) {
        int code = e.getKeyCode();
        if (code >= 0 && code < KeysPressed.length) {
            KeysPressed[code] = true;
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        int code = e.getKeyCode();
        if (code >= 0 && code < KeysPressed.length) {
            KeysPressed[code] = false;
            KeysTyped[code] = true;
        }
    }

    @Override
    public void keyTyped(KeyEvent e) {
        // Intentionally empty; the user of this class can inspect KeysTyped[].
    }

    /**
     * Returns the direction of collision (0 = right, 1 = top, 2 = left, 3 = bottom).
     * stationary - the object we are colliding into
     * projectile - the object that is moving
     * dx = projectile's x displacement
     * dy = projectile's y displacement
     */
    public static int collisionDirection(Rectangle stationary, Rectangle projectile, int dx, int dy) {
        // calculate previous location
        int previousXPos = (int) projectile.getX() - dx;
        int previousYPos = (int) projectile.getY() - dy;
        int height = (int) projectile.getHeight();
        int width = (int) projectile.getWidth();
        int result = 0; // default intersects from right

        if (previousYPos + height <= stationary.getY() && projectile.getMaxY() >= stationary.getY()) {
            // intersects from top
            result = 1;
        } else if (previousXPos + width <= stationary.getX() && projectile.getX() + width >= stationary.getX()) {
            // intersects from left
            result = 2;
        } else if (previousYPos >= stationary.getY() + stationary.height
                && projectile.getY() <= stationary.getY() + stationary.height) {
            // intersects from bottom
            result = 3;
        } else {
            // default/right or fallback
            result = 0;
        }
        return result;
    }

    // --- Getters / Setters ---
    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
        if (frame != null) {
            frame.setTitle(title);
        }
    }

    public void setCleanCanvas(boolean option) {
        this.cleanCanvas = option;
    }

    public static int getMaxWindowX() {
        return MAX_WINDOW_X;
    }

    public static int getMaxWindowY() {
        return MAX_WINDOW_Y;
    }

    public static void setMaxWindowX(int sizeX) {
        MAX_WINDOW_X = sizeX;
    }

    public static void setMaxWindowY(int sizeY) {
        MAX_WINDOW_Y = sizeY;
    }

    public static int getPadding() {
        return PADDING;
    }

    public static void setPadding(int paddingVal) {
        PADDING = paddingVal;
    }

    public void dispose() {
        running = false;
        if (gameThread != null) {
            gameThread.interrupt();
            gameThread = null;
        }
        if (frame != null) {
            frame.dispose();
            frame = null;
        }
    }
}
