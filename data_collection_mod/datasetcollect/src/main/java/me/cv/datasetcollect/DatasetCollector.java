package me.cv.datasetcollect;

import me.voidxwalker.autoreset.Atum;
import me.voidxwalker.autoreset.api.seedprovider.SeedProvider;
import me.voidxwalker.worldpreview.WorldPreview;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.ScreenshotUtils;
import net.minecraft.client.util.Window;
import net.minecraft.text.LiteralText;
import net.minecraft.text.Text;
import net.minecraft.util.math.MathHelper;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Seed queue, capture timing, client settings lock, and screenshot I/O.
 */
public final class DatasetCollector {
    private static final Logger LOGGER = LogManager.getLogger("datasetcollect");

    /** Frames after preview starts (~0.5 s @ 60 FPS). */
    public static final int CAPTURE_FRAME = 30;

    private static final List<Long> seeds = new ArrayList<>();
    private static int currentSeedIndex;
    private static boolean isRunning;
    private static int previewFrameCounter;
    private static boolean capturePending;
    private static Path datasetDir = null;

    private DatasetCollector() {
    }

    public static void loadSeeds(MinecraftClient client) {
        seeds.clear();
        Path seedsFile = client.runDirectory.toPath().resolve("seeds.txt");
        if (!Files.isRegularFile(seedsFile)) {
            LOGGER.warn("seeds.txt not found at {}", seedsFile.toAbsolutePath());
            notify(client, new LiteralText("seeds.txt missing at: " + seedsFile.toAbsolutePath()));
            return;
        }
        try {
            for (String line : Files.readAllLines(seedsFile)) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                String numeric = trimmed.startsWith("-") ? trimmed.substring(1) : trimmed;
                if (numeric.matches("\\d+")) {
                    seeds.add(Long.parseLong(trimmed));
                }
            }
            LOGGER.info("Loaded {} seeds from {}", seeds.size(), seedsFile.toAbsolutePath());
        } catch (IOException e) {
            LOGGER.error("Failed to read seeds.txt", e);
            notify(client, new LiteralText("Failed to read seeds.txt — see log."));
        }
    }

    public static void registerAtumSeedProvider() {
        try {
            Atum.setSeedProvider(new DatasetSeedProvider());
            LOGGER.info("Registered Atum SeedProvider for dataset collection.");
        } catch (IllegalStateException e) {
            LOGGER.error("Could not register SeedProvider — another mod may have claimed it.", e);
        }
    }

    public static boolean hasSeeds() {
        return !seeds.isEmpty();
    }

    public static int getSeedCount() {
        return seeds.size();
    }

    public static boolean isRunning() {
        return isRunning;
    }

    public static int getCurrentSeedIndex() {
        return currentSeedIndex;
    }

    public static List<Long> getSeeds() {
        return Collections.unmodifiableList(seeds);
    }

    public static long getCurrentSeed() {
        if (currentSeedIndex < 0 || currentSeedIndex >= seeds.size()) {
            return 0L;
        }
        return seeds.get(currentSeedIndex);
    }

    public static String getCurrentSeedString() {
        return Long.toString(getCurrentSeed());
    }

    public static Path getDatasetDir(MinecraftClient client) {
        return client.runDirectory.toPath().resolve("dataset");
    }

    public static void start(MinecraftClient client) {
        loadSeeds(client);
        if (seeds.isEmpty()) {
            Path expected = client.runDirectory.toPath().resolve("seeds.txt");
            notify(client, new LiteralText("No seeds loaded. Put seeds.txt here: " + expected.toAbsolutePath()));
            return;
        }
        isRunning = true;
        currentSeedIndex = 0;
        previewFrameCounter = 0;
        capturePending = false;
        datasetDir = getDatasetDir(client);
        try {
            Files.createDirectories(datasetDir);
        } catch (IOException e) {
            LOGGER.error("Failed to create dataset directory at {}", datasetDir.toAbsolutePath(), e);
            notify(client, new LiteralText("Cannot create dataset/ folder — see log."));
            isRunning = false;
            return;
        }
        LOGGER.info("Collection started. Seeds: {}, output: {}", seeds.size(), datasetDir.toAbsolutePath());
        notify(client, new LiteralText("Dataset capture started (" + seeds.size() + " seeds). Saving to dataset/"));
    }

    public static void stop() {
        isRunning = false;
        previewFrameCounter = 0;
        capturePending = false;
        LOGGER.info("Dataset collection stopped.");
    }

    public static void resetPreviewFrameCounter() {
        previewFrameCounter = 0;
        capturePending = false;
    }

    /**
     * Called once per rendered preview frame while on the level loading screen.
     */
    public static void onPreviewFrameRendered(MinecraftClient client) {
        if (!isRunning || !WorldPreview.inPreview() || WorldPreview.isKilled()) {
            return;
        }

        applyClientSettings(client);

        if (capturePending) {
            return;
        }

        previewFrameCounter++;
        if (previewFrameCounter < CAPTURE_FRAME) {
            if (previewFrameCounter == 1) {
                LOGGER.info("Preview active for seed {}, waiting {} frames before capture.", getCurrentSeedString(), CAPTURE_FRAME);
            }
            return;
        }

        capturePending = true;
        long seed = getCurrentSeed();
        try {
            Path saved = saveScreenshot(client, seed);
            recordSpawnCoords(client, seed);
            LOGGER.info("[{}/{}] Saved {}", currentSeedIndex + 1, seeds.size(), saved.toAbsolutePath());
            notify(client, new LiteralText("Saved " + saved.getFileName()));
        } catch (IOException e) {
            LOGGER.error("Screenshot failed for seed {}", seed, e);
            notify(client, new LiteralText("Screenshot failed for seed " + seed + " — see log."));
            capturePending = false;
            return;
        }

        Atum.scheduleReset();
        currentSeedIndex++;
        resetPreviewFrameCounter();

        if (currentSeedIndex >= seeds.size()) {
            LOGGER.info("All {} seeds captured — stopping.", seeds.size());
            notify(client, new LiteralText("Dataset capture finished (" + seeds.size() + " seeds)."));
            stop();
        }
    }

    public static void applyClientSettings(MinecraftClient client) {
        client.options.viewDistance = 5;
        client.options.fov = 110.0;
        client.options.hudHidden = false;
    }

    private static void recordSpawnCoords(MinecraftClient client, long seed) throws IOException {
        if (!WorldPreview.inPreview() || WorldPreview.properties == null) {
            LOGGER.warn("Preview player unavailable — skipped spawn coords for seed {}", seed);
            return;
        }
        int x = MathHelper.floor(WorldPreview.properties.player.getX());
        int z = MathHelper.floor(WorldPreview.properties.player.getZ());

        Path outDir = datasetDir != null ? datasetDir : getDatasetDir(client);
        Path csv = outDir.resolve("spawns.csv");
        boolean writeHeader = !Files.isRegularFile(csv);
        try (BufferedWriter writer = Files.newBufferedWriter(
                csv,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
        )) {
            if (writeHeader) {
                writer.write("seed,x,z\n");
            }
            writer.write(seed + "," + x + "," + z + "\n");
        }
        LOGGER.info("Recorded spawn x={}, z={} for seed {}", x, z, seed);
    }

    private static Path saveScreenshot(MinecraftClient client, long seed) throws IOException {
        Path outDir = datasetDir != null ? datasetDir : getDatasetDir(client);
        Files.createDirectories(outDir);
        Path output = outDir.resolve(seed + ".png");

        Window window = client.getWindow();
        net.minecraft.client.texture.NativeImage image = ScreenshotUtils.takeScreenshot(
                window.getFramebufferWidth(),
                window.getFramebufferHeight(),
                client.getFramebuffer()
        );
        try {
            image.writeFile(output);
        } finally {
            image.close();
        }
        return output;
    }

    public static void notify(MinecraftClient client, Text message) {
        Text prefixed = new LiteralText("[DatasetCollect] ").append(message);
        client.execute(() -> {
            if (client.player != null) {
                client.player.sendMessage(prefixed, false);
            }
        });
    }

    /**
     * Atum official hook — supplies the seed for each reset cycle while collecting.
     */
    private static final class DatasetSeedProvider implements SeedProvider {
        @Override
        public CompletableFuture<String> requestSeed() {
            if (isRunning && currentSeedIndex < seeds.size()) {
                return CompletableFuture.completedFuture(getCurrentSeedString());
            }
            if (Atum.config != null) {
                return CompletableFuture.completedFuture(Atum.config.seed);
            }
            return CompletableFuture.completedFuture("");
        }

        @Override
        public boolean shouldShowSeed() {
            return true;
        }
    }
}
