package me.cv.datasetcollect.mixin;

import me.cv.datasetcollect.DatasetCollector;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.world.ClientWorld;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Re-applies locked video options when the client joins a world.
 */
@Mixin(MinecraftClient.class)
public abstract class MixinMinecraftClient {

    @Inject(method = "joinWorld", at = @At("TAIL"))
    private void datasetcollect$lockSettingsOnJoin(ClientWorld world, CallbackInfo ci) {
        if (DatasetCollector.isRunning()) {
            DatasetCollector.applyClientSettings((MinecraftClient) (Object) this);
        }
    }
}
