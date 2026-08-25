package utilities;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public abstract class GDV5 extends JPanel implements Runnable, KeyListener {
    public static final int WIDTH = 1200;
    public static final int HEIGHT = 800;
    public static final boolean[] KeysPressed = new boolean[256];

    private Thread gameThread;
    private volatile boolean running;

    public GDV5() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(Color.BLACK);
        setFocusable(true);
        addKeyListener(this);
    }

    public abstract void update();

    public abstract void draw(Graphics2D win);

    public void start() {
        if (running) {
            return;
        }
        running = true;

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Snake Game");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setContentPane(this);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            requestFocusInWindow();
        });

        gameThread = new Thread(this, "GDV5-Loop");
        gameThread.start();
    }

    @Override
    public void run() {
        final long frameTimeNanos = 1_000_000_000L / 60L;
        while (running) {
            long start = System.nanoTime();
            update();
            repaint();
            long elapsed = System.nanoTime() - start;
            long sleepNanos = frameTimeNanos - elapsed;
            if (sleepNanos > 0) {
                try {
                    Thread.sleep(sleepNanos / 1_000_000L, (int) (sleepNanos % 1_000_000L));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    running = false;
                }
            }
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        draw(g2);
        g2.dispose();
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
        }
    }

    @Override
    public void keyTyped(KeyEvent e) {
    }
}
