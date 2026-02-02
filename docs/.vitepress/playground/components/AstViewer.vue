<script setup lang="ts">
interface AstNode {
  type: string
  value?: string | number
  children?: AstNode[]
}

defineProps<{
  ast: AstNode
}>()
</script>

<template>
  <div class="ast-viewer">
    <AstNodeView :node="ast" :depth="0" />
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed } from 'vue'

export default defineComponent({ name: 'AstViewer' })

const AstNodeView = defineComponent({
  name: 'AstNodeView',
  props: {
    node: { type: Object as () => { type: string; value?: string | number; children?: any[] }, required: true },
    depth: { type: Number, required: true },
  },
  setup(props) {
    const expanded = ref(props.depth < 3)
    const hasChildren = computed(() => props.node.children && props.node.children.length > 0)
    const toggle = () => {
      if (hasChildren.value) expanded.value = !expanded.value
    }
    const displayValue = computed(() => {
      if (props.node.value === undefined) return null
      return typeof props.node.value === 'string' ? `"${props.node.value}"` : props.node.value
    })

    return { expanded, hasChildren, toggle, displayValue }
  },
  template: `
    <div class="ast-node" :style="{ marginLeft: depth > 0 ? '16px' : '0' }">
      <div class="ast-node__header" @click="toggle">
        <span class="ast-node__toggle">
          <template v-if="hasChildren">{{ expanded ? '\u25BC' : '\u25B6' }}</template>
        </span>
        <span class="ast-node__type">{{ node.type }}</span>
        <span v-if="displayValue !== null" class="ast-node__value">{{ displayValue }}</span>
      </div>
      <div v-if="hasChildren && expanded" class="ast-node__children">
        <AstNodeView v-for="(child, i) in node.children" :key="i" :node="child" :depth="depth + 1" />
      </div>
    </div>
  `,
})
</script>

<style scoped>
.ast-viewer {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.6;
}

.ast-node__header {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 1px 0;
  border-radius: 3px;
}

.ast-node__header:hover {
  background: var(--vp-c-bg-soft);
}

.ast-node__toggle {
  width: 14px;
  font-size: 10px;
  text-align: center;
  color: var(--vp-c-text-3);
  flex-shrink: 0;
}

.ast-node__type {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.ast-node__value {
  color: var(--vp-c-text-2);
}

.ast-node__children {
  border-left: 1px solid var(--vp-c-divider);
  margin-left: 6px;
}
</style>
