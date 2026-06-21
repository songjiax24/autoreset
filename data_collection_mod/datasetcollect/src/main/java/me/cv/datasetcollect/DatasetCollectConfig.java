package me.cv.datasetcollect;

import net.minecraft.client.MinecraftClient;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Optional game-directory config: {@code datasetcollect.cfg}
 *
 * <pre>
 * auto_start=true
 * start_delay_seconds=8
 * </pre>
 */
public final class DatasetCollectConfig {
    private static final Logger LOGGER = LogManager.getLogger("datasetcollect");

    public boolean autoStart = true;
    public int startDelaySeconds = 8;

    public static DatasetCollectConfig load(MinecraftClient client) {
        DatasetCollectConfig config = new DatasetCollectConfig();
        Path cfgFile = client.runDirectory.toPath().resolve("datasetcollect.cfg");
        if (!Files.isRegularFile(cfgFile)) {
            LOGGER.info("No datasetcollect.cfg — using defaults (auto_start={}, delay={}s).",
                    config.autoStart, config.startDelaySeconds);
            return config;
        }
        try {
            List<String> lines = Files.readAllLines(cfgFile);
            for (String rawLine : lines) {
                String line = rawLine.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                int eq = line.indexOf('=');
                if (eq <= 0) {
                    continue;
                }
                String key = line.substring(0, eq).trim().toLowerCase(Locale.ROOT);
                String value = line.substring(eq + 1).trim();
                switch (key) {
                    case "auto_start":
                        config.autoStart = parseBoolean(value, config.autoStart);
                        break;
                    case "start_delay_seconds":
                    case "delay_seconds":
                        config.startDelaySeconds = parsePositiveInt(value, config.startDelaySeconds);
                        break;
                    default:
                        LOGGER.warn("Unknown datasetcollect.cfg key: {}", key);
                        break;
                }
            }
            LOGGER.info("Loaded datasetcollect.cfg (auto_start={}, delay={}s).",
                    config.autoStart, config.startDelaySeconds);
        } catch (IOException e) {
            LOGGER.error("Failed to read datasetcollect.cfg — using defaults.", e);
        }
        return config;
    }

    private static boolean parseBoolean(String value, boolean fallback) {
        if ("true".equalsIgnoreCase(value) || "yes".equalsIgnoreCase(value) || "1".equals(value)) {
            return true;
        }
        if ("false".equalsIgnoreCase(value) || "no".equalsIgnoreCase(value) || "0".equals(value)) {
            return false;
        }
        return fallback;
    }

    private static int parsePositiveInt(String value, int fallback) {
        try {
            int parsed = Integer.parseInt(value);
            return parsed > 0 ? parsed : fallback;
        } catch (NumberFormatException e) {
            return fallback;
        }
    }
}
