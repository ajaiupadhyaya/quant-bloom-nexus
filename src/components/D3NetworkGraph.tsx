import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface NetworkNode extends d3.SimulationNodeDatum {
  id: string;
  group: number;
  size?: number;
  label?: string;
}

interface NetworkLink {
  source: string;
  target: string;
  value: number;
  type?: string;
}

interface D3NetworkGraphProps {
  nodes: NetworkNode[];
  links: NetworkLink[];
  width?: number;
  height?: number;
  colors?: string[];
  title?: string;
  style?: React.CSSProperties;
}

export const D3NetworkGraph: React.FC<D3NetworkGraphProps> = ({
  nodes,
  links,
  width = 800,
  height = 600,
  colors = ['#ff6b35', '#00d4ff', '#00ff88', '#ff4757', '#ffa500'],
  title,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!nodes || !links || nodes.length === 0) return;

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Create filters and gradients
    const defs = svg.append('defs');
    
    // Glow filter
    const glowFilter = defs.append('filter')
      .attr('id', 'network-glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Link gradient
    const linkGradient = defs.append('linearGradient')
      .attr('id', 'link-gradient')
      .attr('gradientUnits', 'userSpaceOnUse');

    linkGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', colors[1])
      .attr('stop-opacity', 0.8);

    linkGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', colors[1])
      .attr('stop-opacity', 0.2);

    // Color scale for groups
    const colorScale = d3.scaleOrdinal()
      .domain(nodes.map(d => d.group.toString()))
      .range(colors);

    // Size scale for nodes
    const sizeScale = d3.scaleLinear()
      .domain(d3.extent(nodes, d => d.size || 1) as [number, number])
      .range([5, 25]);

    // Link width scale
    const linkWidthScale = d3.scaleLinear()
      .domain(d3.extent(links, d => d.value) as [number, number])
      .range([1, 8]);

    // Create simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => sizeScale(d.size || 1) + 5));

    // Tooltip
    const tooltip = d3.select(ref.current?.parentElement)
      .selectAll('.network-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'network-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('border', '1px solid #ff6b35')
      .style('border-radius', '6px')
      .style('padding', '12px')
      .style('font-size', '12px')
      .style('color', '#ffffff')
      .style('box-shadow', '0 4px 20px rgba(255, 107, 53, 0.3)')
      .style('backdrop-filter', 'blur(10px)')
      .style('display', 'none');

    // Create links
    const link = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => `url(#link-gradient)`)
      .attr('stroke-width', d => linkWidthScale(d.value))
      .attr('opacity', 0.6)
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('opacity', 1)
          .attr('filter', 'url(#network-glow)');

        tooltip
          .style('display', 'block')
          .html(`
            <div style="border-bottom: 1px solid #ff6b35; margin-bottom: 8px; padding-bottom: 8px;">
              <strong style="color: #ff6b35;">Connection</strong>
            </div>
            <div style="margin-bottom: 4px;"><strong>From:</strong> ${d.source}</div>
            <div style="margin-bottom: 4px;"><strong>To:</strong> ${d.target}</div>
            <div style="margin-bottom: 4px;"><strong>Strength:</strong> ${d.value.toFixed(2)}</div>
            ${d.type ? `<div><strong>Type:</strong> ${d.type}</div>` : ''}
          `)
          .style('left', (event.pageX + 15) + 'px')
          .style('top', (event.pageY - 15) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('opacity', 0.6)
          .attr('filter', null);

        tooltip.style('display', 'none');
      });

    // Create nodes
    const node = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => sizeScale(d.size || 1))
      .attr('fill', d => colorScale(d.group.toString()) as string)
      .attr('stroke', '#000')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, NetworkNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      )
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', sizeScale(d.size || 1) * 1.5)
          .attr('filter', 'url(#network-glow)');

        // Highlight connected links
        link.attr('opacity', l => 
          ((l.source as any).id === d.id || (l.target as any).id === d.id) ? 1 : 0.1
        );

        tooltip
          .style('display', 'block')
          .html(`
            <div style="border-bottom: 1px solid #ff6b35; margin-bottom: 8px; padding-bottom: 8px;">
              <strong style="color: #ff6b35;">${d.label || d.id}</strong>
            </div>
            <div style="margin-bottom: 4px;"><strong>ID:</strong> ${d.id}</div>
            <div style="margin-bottom: 4px;"><strong>Group:</strong> ${d.group}</div>
            <div style="margin-bottom: 4px;"><strong>Size:</strong> ${d.size || 1}</div>
            <div style="color: #00d4ff;"><strong>Connections:</strong> ${links.filter(l => l.source === d.id || l.target === d.id).length}</div>
          `)
          .style('left', (event.pageX + 15) + 'px')
          .style('top', (event.pageY - 15) + 'px');
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', sizeScale(d.size || 1))
          .attr('filter', null);

        link.attr('opacity', 0.6);
        tooltip.style('display', 'none');
      });

    // Add labels
    const label = svg.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text(d => d.label || d.id)
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', '#fff')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .style('pointer-events', 'none');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Chart title
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', colors[0])
        .text(title);
    }

    // Legend
    const uniqueGroups = Array.from(new Set(nodes.map(d => d.group)));
    const legend = svg.append('g')
      .attr('transform', `translate(20, 60)`);

    uniqueGroups.forEach((group, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 25})`);

      legendItem.append('circle')
        .attr('r', 8)
        .attr('fill', colorScale(group.toString()) as string)
        .attr('stroke', '#000')
        .attr('stroke-width', 1);

      legendItem.append('text')
        .attr('x', 20)
        .attr('y', 5)
        .attr('font-size', '12px')
        .attr('fill', '#888')
        .text(`Group ${group}`);
    });

    // Controls info
    const controls = svg.append('g')
      .attr('transform', `translate(20, ${height - 60})`);

    controls.append('text')
      .attr('font-size', '11px')
      .attr('fill', '#666')
      .text('• Drag nodes to move them');

    controls.append('text')
      .attr('y', 15)
      .attr('font-size', '11px')
      .attr('fill', '#666')
      .text('• Hover for details');

  }, [nodes, links, width, height, colors, title]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg 
        ref={ref} 
        width={width} 
        height={height}
        style={{ 
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
          borderRadius: '4px'
        }}
      />
    </div>
  );
};