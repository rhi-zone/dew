<script setup lang="ts">
defineProps<{
  tabs: string[]
  activeTab: string
  outputs: Record<string, { ok: boolean; code?: string; error?: string }>
}>()

defineEmits<{
  'update:activeTab': [tab: string]
}>()
</script>

<template>
  <div class="output-panel">
    <div class="output-panel__tabs">
      <button
        v-for="tab in tabs"
        :key="tab"
        class="output-panel__tab"
        :class="{ 'output-panel__tab--active': tab === activeTab }"
        @click="$emit('update:activeTab', tab)"
      >
        {{ tab }}
      </button>
    </div>
    <div class="output-panel__content">
      <template v-for="tab in tabs" :key="tab">
        <div v-if="tab === activeTab" class="output-panel__pane">
          <template v-if="outputs[tab]">
            <pre v-if="outputs[tab].ok" class="output-panel__code">{{ outputs[tab].code }}</pre>
            <div v-else class="output-panel__error">{{ outputs[tab].error }}</div>
          </template>
          <div v-else class="output-panel__placeholder">WASM not loaded</div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.output-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.output-panel__tabs {
  display: flex;
  gap: 2px;
  padding: 8px 12px 0;
  border-bottom: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}

.output-panel__tab {
  padding: 6px 14px;
  font-size: 13px;
  font-family: var(--vp-font-family-mono);
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s;
  margin-bottom: -1px;
}

.output-panel__tab:hover {
  color: var(--vp-c-text-1);
}

.output-panel__tab--active {
  color: var(--vp-c-brand-1);
  border-bottom-color: var(--vp-c-brand-1);
}

.output-panel__content {
  flex: 1;
  overflow: auto;
  min-height: 0;
}

.output-panel__pane {
  padding: 12px;
  height: 100%;
}

.output-panel__code {
  margin: 0;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-1);
  white-space: pre-wrap;
  word-break: break-word;
}

.output-panel__error {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  color: var(--vp-c-danger-1);
}

.output-panel__placeholder {
  font-size: 13px;
  color: var(--vp-c-text-3);
  font-style: italic;
}
</style>
